# model.jl — the model constructor.
#
# ExaCUTEstModel(name) decodes the SIF file with ExaModelsSIF and returns an
# ExaModel: an AbstractNLPModel, so the whole NLPModels API (obj, grad, cons,
# jac_coord, hess_coord, jprod, hprod, ...) works directly, as it does on a
# CUTEstModel. Unlike CUTEstModel there is no Fortran decode-and-compile step,
# no shared library on disk, no global decoder state, and therefore no
# requirement to finalize one model before building another: models are
# independent Julia objects and any number can coexist.

"""
    ExaCUTEstModel(name, args...; T = Float64, backend = nothing,
                   params = Dict(), kwargs...)

Load SIF problem `name` from the active collection (see [`set_mastsif`](@ref))
as an `ExaModel`. `name` may also be a path to a `.SIF` file.

# Arguments
- `name`: problem name, e.g. `"ROSENBR"`, or a path to a SIF file.
- `args...`: `sifdecoder`-style parameter overrides, accepted in CUTEst.jl's
  form so existing call sites port unchanged:
  `ExaCUTEstModel("READING4", "-param", "N=100")`.

# Keyword arguments
- `T`: element type (`Float64` default; `Float32` and `BigFloat` work).
- `backend`: an ExaModels/KernelAbstractions backend, e.g. `CUDABackend()`, to
  build the model on a GPU. This is the capability CUTEst.jl cannot offer: the
  problem becomes a SIMD-abstracted ExaModel rather than Fortran callbacks.
- `params`: parameter overrides as a `Dict`, e.g. `Dict("N" => 100)`.
  Equivalent to `args...`; merged, with `params` winning on a clash.
- `prod`: whether a decoded or native model carries the product helper that
  `hprod!`, `jprod_nln!` and `jtprod_nln!` dispatch on under a `backend`.
  `:auto` (the builders' default) builds it wherever the model's structure
  admits one; see `ExaModelsSIF.prodflag`.
- `kwargs...`: forwarded to `ExaModelsSIF.sif_decode`, or to the native
  builder for a problem in [`native_problems`](@ref).

Problems whose SIF file appends external Fortran are built from their native
Julia implementations (see [`external_fortran_problems`](@ref)). Such a
problem with `\$-PARAMETER` overrides, or a SIF file outside the collection
that appends external Fortran, raises `ExaModelsSIF.ExternalFortranError`.

# Examples
```julia
nlp = ExaCUTEstModel("ROSENBR")
obj(nlp, nlp.meta.x0)

nlp = ExaCUTEstModel("READING4", "-param", "N=100")   # size parameter
nlp = ExaCUTEstModel("FMINSURF"; params = Dict("P" => 32))
nlp = ExaCUTEstModel("ROSENBR"; backend = CUDABackend())   # on the GPU
```
"""
function ExaCUTEstModel(name::AbstractString, args...; T = Float64,
                        backend = nothing, params::AbstractDict = Dict{String,Any}(),
                        kwargs...)
    path = occursin(".SIF", name) && isfile(name) ? abspath(name) : sif_problem_path(name)
    allparams = merge(_parse_param_args(args), Dict{String,Any}(params))
    # Only external-Fortran problems have native entries, so this cannot shadow
    # a decodable problem. `kwargs` reach the native model too: several take
    # options of their own (`element_hessian`, `ref_param_rounding`, `single`,
    # `fortran_literals`) that select between readings of the Fortran, and a
    # keyword a native model does not accept raises a MethodError naming it.
    if isempty(allparams) && haskey(NATIVE_MODELS, String(name))
        return NATIVE_MODELS[String(name)](; T, backend, kwargs...)
    end
    return ExaModelsSIF.sif_model(path; T, backend, params = allparams, kwargs...)
end

"""
    _parse_param_args(args) -> Dict{String,Any}

Accept CUTEst.jl's `sifdecoder` argument style for `\$-PARAMETER` overrides and
return them as a Dict. Recognized forms, with or without a preceding
`"-param"` token: `"-param", "N=100"`, `"N=100"`, `"-param", "N", "100"`.
Values parse as Int where possible, else Float64, else stay strings.
"""
function _parse_param_args(args)
    out = Dict{String,Any}()
    toks = String[string(a) for a in args]
    i = 1
    while i <= length(toks)
        t = toks[i]
        if t == "-param" || t == "--param"
            i += 1
            continue
        end
        if occursin('=', t)
            k, v = split(t, '=', limit = 2)
            out[strip(k)] = _parse_param_value(strip(v))
            i += 1
        elseif i < length(toks)
            out[strip(t)] = _parse_param_value(strip(toks[i+1]))
            i += 2
        else
            error("unpaired sifdecoder argument \"$t\"; expected NAME=VALUE or NAME VALUE")
        end
    end
    return out
end

function _parse_param_value(s::AbstractString)
    v = tryparse(Int, s)
    v === nothing || return v
    v = tryparse(Float64, replace(s, r"[dD](?=[0-9+-])" => "e"))
    v === nothing || return v
    return String(s)
end

"""
    sif_variable_names(name_or_path; kwargs...) -> Vector{String}
    sif_constraint_names(name_or_path; kwargs...) -> Vector{String}

The SIF names of a problem's variables and of its constraint groups, in the
order the model's vectors use. CUTEst.jl reorders both (nonlinear variables
first; equality and linear rows first) while a decoded model keeps SIF
declaration order, so these are what map one indexing onto the other.
"""
function sif_variable_names(name::AbstractString; kwargs...)
    P = _decode(name; kwargs...)
    return copy(P.varnames)
end

function sif_constraint_names(name::AbstractString; kwargs...)
    P = _decode(name; kwargs...)
    return [gn for gn in P.grouporder if P.groups[gn].kind != 'N']
end

"""
    row_linearity(name; kwargs...) -> (nlin = Int, nnln = Int)

How many of a problem's constraint rows are linear and how many nonlinear.

Read from the decoded SIF rather than from the model's `meta`: ExaModels does
not classify row linearity and reports `nlin = 0, nnln = ncon` for every
model, so `nlp.meta.nlin` is not the answer (measured on HS14 and HS35, which
have linear rows reported as nonlinear). A group is linear exactly when it
has no nonlinear elements and no nontrivial group function.
"""
function row_linearity(name::AbstractString; kwargs...)
    P = _decode(name; kwargs...)
    nlin = 0
    nnln = 0
    for gn in P.grouporder
        g = P.groups[gn]
        g.kind == 'N' && continue
        islin = isempty(g.elems) && ExaModelsSIF.gtypename(P, g) === nothing
        islin ? (nlin += 1) : (nnln += 1)
    end
    return (nlin = nlin, nnln = nnln)
end

# `external = :defer` rather than ExaModelsSIF's `:error` default, because
# every caller above asks a STRUCTURAL question -- which variables, which
# groups, which groups are linear -- and none of those readings depends on
# what an element function computes. Deferring makes them answer for all 1548
# problems instead of the 1514 with no external Fortran; with `:error` they
# throw on the 34, which is the wrong answer to "what are this problem's
# variables called" and, worse, breaks the name-permutation step that every
# verification of a ported model needs.
function _decode(name::AbstractString; params::AbstractDict = Dict{String,Any}(),
                 external::Symbol = :defer, kwargs...)
    path = occursin(".SIF", name) && isfile(name) ? abspath(name) : sif_problem_path(name)
    return ExaModelsSIF.sif_decode(path; params = Dict{String,Any}(params),
                                   external, kwargs...)
end

# NOTE on `finalize`: a CUTEstModel must be finalized before another with the
# same name is built, because the Fortran decoder keeps global state in a
# shared library. A decoded ExaModel keeps none, so ported code that calls
# `finalize(nlp)` needs no shim -- Base.finalize already accepts an object
# with no finalizers registered (measured, not assumed) -- and code that
# forgets to call it is simply correct here.
