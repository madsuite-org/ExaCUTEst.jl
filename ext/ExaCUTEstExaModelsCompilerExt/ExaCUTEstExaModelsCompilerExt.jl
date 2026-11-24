# Compiling CUTEst problems into one shared library.
#
# A CUTEst problem is fixed by its SIF file: its sizes are SIF parameters, decoded
# into the model, so each problem is compiled as a fixed model at its default
# parameters (or at the ones given in `params`). Every problem has its own types,
# so the cost of a library grows with the number of problems in it; `problems`
# picks them, by name or with `select_sif_problems`. The default is `library.txt` here,
# the problems known to compile and evaluate; `chunk = (k, N)` compiles the k-th of N
# parts of the list, which cnlplib's library workflow builds in parallel and joins.
#
# Only problems the SIF decoder reads are compiled. The 34 problems with external
# Fortran are native Julia models here (`native_problems()`), several of them
# oracles that the compiler does not take, so they are left out. With `device`, so are
# problems whose cores carry buffered subexpressions.
module ExaCUTEstExaModelsCompilerExt

import ExaCUTEst
import ExaModelsCompiler
import ExaModelsSIF
using ExaModelsCompiler: compile_library, select

# A CUTEst name is not always a C identifier (`BA-L1`, `10FOLDTR`); the prefix is the
# name in lowercase with every other character replaced by `_`, and `_` in front of a
# leading digit (`rosenbr`, `ba_l1`, `_10foldtr`).
function _prefix(name::AbstractString)
    s = replace(lowercase(name), r"[^a-z0-9_]" => "_")
    return Symbol(isdigit(first(s)) ? "_" * s : s)
end

const LIBRARY = filter(l -> !isempty(l) && !startswith(l, '#'),
                       strip.(readlines(joinpath(@__DIR__, "library.txt"))))

"""
    compile_all(ExaCUTEst; path = "@cutest", problems, params = Dict(), T = Float64,
                chunk = nothing, only = nothing, exclude = (), kwargs...)

Compile the CUTEst problems named in `problems` (default: the list in `library.txt`)
into one shared library, each as a fixed model under its name in lowercase with
non-identifier characters replaced by `_`. `params` gives SIF parameters per problem,
`Dict("DIXMAANA1" => Dict("M" => 1000))`. Problems in `native_problems()` are refused.
`chunk = (k, N)` compiles only the k-th (from 0) of N interleaved parts of the sorted
list.
"""
function ExaModelsCompiler.compile_all(
    ::Val{ExaCUTEst};
    path = "@cutest",
    problems = LIBRARY,
    params::AbstractDict = Dict{String, Any}(),
    T = Float64,
    chunk = nothing,
    only = nothing,
    exclude = (),
    kwargs...,
)
    native = Set(ExaCUTEst.native_problems())
    names = sort!(unique(String.(problems)))
    if chunk !== nothing
        k, n = chunk
        0 <= k < n || throw(ArgumentError("chunk = ($k, $n): need 0 <= k < N"))
        names = names[(k + 1):n:end]   # interleaved, so large and small problems spread out
    end
    bad = filter(in(native), names)
    isempty(bad) || throw(ArgumentError("native models cannot be compiled: $(join(bad, ", "))"))
    pairs = [_prefix(n) => n for n in names]
    allunique(first.(pairs)) || throw(ArgumentError("two problems map to one prefix"))
    # Filter before decoding: decoding is the expensive part of assembling here.
    kept = Set(first.(select([p => nothing for p in first.(pairs)]; only, exclude)))
    models = map(filter(p -> first(p) in kept, pairs)) do (prefix, name)
        p = Dict{String, Any}(get(params, name, Dict{String, Any}()))
        prefix => ExaModelsSIF.sif_core(ExaCUTEst.sif_problem_path(name); T, params = p)
    end
    # The device tier does not take buffered subexpressions, which the decoder emits
    # for wide nontrivial groups; such problems are left out of a device library.
    if get(kwargs, :device, :cpu) !== :cpu
        buffered = [first(m) for m in models if !isempty(last(m).subexprs)]
        isempty(buffered) || @warn "left out of the device library (buffered subexpressions): $(join(buffered, ", "))"
        models = [m for m in models if isempty(last(m).subexprs)]
    end
    return compile_library(path, models...; kwargs...)
end

end # module
