# Record the ExaModels block type signatures of every model, for plan.py.
#
#   CV_RUN=<run dir> SIGSCAN_LIST=<problem list> SIGSCAN_OUT=<tsv> \
#       julia --project=test/gpu test/collection/sigscan.jl
#
# (sigscan.sh drives this over the whole population.) Each ExaModel carries
# its objective and constraint blocks in `m.objs` / `m.cons`; the
# KernelAbstractions kernels are generic over a block's `f` (the SIMD AD
# function) and the element type of its iterator, so one distinct
# `(typeof(f), eltype(itr))` pair is one distinct kernel specialization per
# kernel that block needs. Those two types do not depend on the backend, so the
# signatures are collected from CPU-built models, with no kernel compiled.
#
# Signatures are keyed by a SHA1 of their full serialized type, which is stable
# across processes. Restartable: appends to SIGSCAN_OUT and skips problems
# already present, exiting with status 42 after SIGSCAN_CHUNK problems so the
# driver can start a fresh process.
#
# Output columns: problem, nvar, ncon, kind (obj | con | none | ERROR | KILLED),
# digest, signature length, signature text (truncated).
include(joinpath(@__DIR__, "..", "gpu", "activate.jl"))
using ExaCUTEst, SHA

# A type's full structure, serialized without `show`: ExaModels abbreviates
# its AD node types in `show` (`Node2{*,…}`), and that abbreviated part is what
# distinguishes one kernel specialization from another.
function typesig_emit(io::IO, @nospecialize(p))
    if p isa DataType
        print(io, parentmodule(p), '.', nameof(p))
        if !isempty(p.parameters)
            print(io, '{')
            for (i, q) in enumerate(p.parameters)
                i > 1 && print(io, ',')
                typesig_emit(io, q)
            end
            print(io, '}')
        end
    elseif p isa UnionAll
        print(io, "UnionAll(")
        typesig_emit(io, p.var)
        print(io, ',')
        typesig_emit(io, p.body)
        print(io, ')')
    elseif p isa Union
        print(io, "Union(")
        for (i, q) in enumerate(Base.uniontypes(p))
            i > 1 && print(io, ',')
            typesig_emit(io, q)
        end
        print(io, ')')
    elseif p isa TypeVar
        print(io, "TV(", p.name, ')')
    elseif p isa Core.TypeofBottom
        print(io, "Union{}")
    elseif p isa Tuple                 # e.g. the field-name tuple of a NamedTuple
        print(io, '(')
        for (i, q) in enumerate(p)
            i > 1 && print(io, ',')
            typesig_emit(io, q)
        end
        print(io, ')')
    elseif p isa Symbol || p isa Integer || p isa Bool || p isa Char
        print(io, repr(p))
    else
        print(io, "val<", nameof(typeof(p)), ':', repr(p), '>')
    end
    return nothing
end
typesig(@nospecialize(T)) = sprint(typesig_emit, T)

const OUT = ENV["SIGSCAN_OUT"]
const LIST = ENV["SIGSCAN_LIST"]
const CUR = OUT * ".current"
const CHUNK = parse(Int, get(ENV, "SIGSCAN_CHUNK", "40"))
const TEXTCAP = 4000

done = Set{String}()
isfile(OUT) && for ln in eachline(OUT)
    push!(done, String(first(split(ln, '\t'))))
end
# A problem in flight when the process died is recorded, not retried, so one
# wedged decode costs one problem.
if isfile(CUR)
    prev = strip(read(CUR, String))
    if !isempty(prev) && !(prev in done)
        open(io -> println(io, join([prev, "-1", "-1", "KILLED", "-", "0", "-"], '\t')), OUT, "a")
        push!(done, String(prev))
    end
end

# `eltype(itr)` rather than `typeof(itr)`: the container differs between a host
# and a device build while the kernel specialization does not.
blocksig(b) = string(typesig(typeof(b.f)), " |ITR| ", typesig(eltype(b.itr)))
digest(s) = bytes2hex(SHA.sha1(s))[1:16]

names = [strip(ln) for ln in eachline(LIST) if !isempty(strip(ln))]
nproc = 0
open(OUT, "a") do io
    for name in names
        name in done && continue
        write(CUR, name)
        try
            m = ExaCUTEstModel(name)
            nv, nc = m.meta.nvar, m.meta.ncon
            for (kind, blocks) in (("obj", m.objs), ("con", m.cons)), b in blocks
                s = blocksig(b)
                txt = replace(first(s, TEXTCAP), '\t' => ' ', '\n' => ' ')
                println(io, join([name, nv, nc, kind, digest(s), length(s), txt], '\t'))
            end
            isempty(m.objs) && isempty(m.cons) &&
                println(io, join([name, nv, nc, "none", "-", "0", "-"], '\t'))
        catch err
            println(io, join([name, "-1", "-1", "ERROR", "-", "0",
                replace(first(sprint(showerror, err), 120), '\t' => ' ', '\n' => ' ')], '\t'))
        end
        flush(io)
        rm(CUR; force = true)
        global nproc += 1
        nproc >= CHUNK && exit(42)
    end
end
println("sigscan complete")
