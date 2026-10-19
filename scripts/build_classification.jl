# scripts/build_classification.jl — regenerate src/classf.json.
#
# The classification LETTERS come from CLASSF.DB (shipped in the collection
# artifact); the SIZES and bound/row-type COUNTS come from decoding every
# problem, because CLASSF.DB records "V" wherever a problem's size is a
# $-PARAMETER. CUTEst.jl builds its classf.json the same way, via the Fortran
# decoder; this does it with ExaModelsSIF, so regenerating needs no Fortran.
#
#   julia --project=. scripts/build_classification.jl
#
# External-Fortran problems have no decode, so their sizes come from a
# CUTEstModel. The script therefore runs in the test/gpu environment, which
# has CUTEst.jl, and switches to it itself.
include(joinpath(@__DIR__, "..", "test", "gpu", "activate.jl"))
using ExaCUTEst, ExaModelsSIF, NLPModels, JSON
import CUTEst
using FileWatching.Pidfile: mkpidlock

dir = ExaCUTEst.mastsif()
db = joinpath(dir, "CLASSF.DB")
isfile(db) || error("CLASSF.DB not found in $dir")

entries = Dict{String,Any}()
nfail = 0

# Bound/row classes exactly as NLPModelMeta defines them: fixed (equal
# bounds), free (both infinite), bounded below only, above only, both.
function varclasses(lvar, uvar)
    fixed = free = below = above = both = 0
    for (l, u) in zip(lvar, uvar)
        if l == u
            fixed += 1
        elseif isinf(l) && isinf(u)
            free += 1
        elseif isinf(u)
            below += 1
        elseif isinf(l)
            above += 1
        else
            both += 1
        end
    end
    return (; fixed, free, below, above, both)
end

# Constraint bounds from the decoded groups. Mirrors builder.jl's rules (group
# constant folded into the row value, so bounds sit around 0; a RANGES value
# widens one side; |bound| >= 1e20 is infinite). The sanity check below
# confirms this agrees with a fully built model, which is the authority.
function conbounds(P)
    lcon = Float64[]
    ucon = Float64[]
    for gn in P.grouporder
        g = P.groups[gn]
        g.kind == 'N' && continue
        triv = ExaModelsSIF.gtypename(P, g) === nothing
        r = g.range_ === nothing ? nothing : (triv ? g.range_ / g.scale : g.range_)
        if g.kind == 'E'
            r === nothing ? (push!(lcon, 0.0); push!(ucon, 0.0)) :
                r >= 0 ? (push!(lcon, 0.0); push!(ucon, r)) :
                         (push!(lcon, r); push!(ucon, 0.0))
        elseif g.kind == 'L'
            push!(lcon, r === nothing ? -Inf : -abs(r)); push!(ucon, 0.0)
        elseif g.kind == 'G'
            push!(lcon, 0.0); push!(ucon, r === nothing ? Inf : abs(r))
        else
            error("unexpected constraint kind $(g.kind) on $gn")
        end
    end
    for i in eachindex(lcon)
        lcon[i] <= -1e20 && (lcon[i] = -Inf)
        ucon[i] >= 1e20 && (ucon[i] = Inf)
    end
    return lcon, ucon
end

function decoded_entry(name, dir, toks, cls)
    P = ExaModelsSIF.sif_decode(joinpath(dir, name * ".SIF"))
    v = varclasses(P.lvar, P.uvar)
    lcon, ucon = conbounds(P)
    c = varclasses(lcon, ucon)
    nlin = nnln = 0
    for gn in P.grouporder
        g = P.groups[gn]
        g.kind == 'N' && continue
        isempty(g.elems) && ExaModelsSIF.gtypename(P, g) === nothing ?
            (nlin += 1) : (nnln += 1)
    end
    return Dict(
        "objtype" => cls.objtype,
        "contype" => cls.contype,
        "regular" => cls.regular,
        "derivative_order" => cls.derivative_order,
        "origin" => cls.origin,
        "has_internal_var" => cls.has_internal_var,
        "variables" => Dict(
            "can_choose" => first(toks[3]) == 'V',
            "number" => length(P.varnames),
            "fixed" => v.fixed, "free" => v.free,
            "bounded_below" => v.below, "bounded_above" => v.above,
            "bounded_both" => v.both,
        ),
        "constraints" => Dict(
            "can_choose" => first(toks[4]) == 'V',
            "number" => length(lcon),
            "equality" => c.fixed,
            "ineq_below" => c.below, "ineq_above" => c.above,
            "ineq_both" => c.both,
            # NOT meta.nlin/nnln: ExaModels reports every row nonlinear
            "linear" => nlin, "nonlinear" => nnln,
        ),
    )
end

# External-Fortran problems have no decode, so their metadata comes from the
# compiled Fortran instead, read from a CUTEstModel, which classifies row
# linearity (HS85 has 1 linear row, SAROMM 1095).
function fortran_entry(name, toks, cls)
    # Finalize: the Fortran decoder keeps global state in a shared library, so
    # a model left open makes later constructions fail. CUTEst also compiles
    # every problem into one shared directory, so the build holds the lock.
    ref = mkpidlock(() -> CUTEst.CUTEstModel(name), "/tmp/cutest-decode.pid"; wait = true)
    m = ref.meta
    counts = (nvar = m.nvar, ncon = m.ncon,
              ifix = length(m.ifix), ifree = length(m.ifree),
              ilow = length(m.ilow), iupp = length(m.iupp), irng = length(m.irng),
              jfix = length(m.jfix), jlow = length(m.jlow),
              jupp = length(m.jupp), jrng = length(m.jrng),
              nlin = m.nlin, nnln = m.nnln)
    finalize(ref)
    return Dict(
        "objtype" => cls.objtype, "contype" => cls.contype,
        "regular" => cls.regular, "derivative_order" => cls.derivative_order,
        "origin" => cls.origin, "has_internal_var" => cls.has_internal_var,
        "variables" => Dict(
            "can_choose" => first(toks[3]) == 'V', "number" => counts.nvar,
            "fixed" => counts.ifix, "free" => counts.ifree,
            "bounded_below" => counts.ilow, "bounded_above" => counts.iupp,
            "bounded_both" => counts.irng,
        ),
        "constraints" => Dict(
            "can_choose" => first(toks[4]) == 'V', "number" => counts.ncon,
            "equality" => counts.jfix, "ineq_below" => counts.jlow,
            "ineq_above" => counts.jupp, "ineq_both" => counts.jrng,
            "linear" => counts.nlin, "nonlinear" => counts.nnln,
        ),
    )
end

# CLASSF.DB plus the problems missing from it. Nine problems in the current
# collection (the BA-L16/BA-L52 pairs, LRCOVTYPE, the four MNISTS*) load fine
# but have no CLASSF.DB line, so taking the DB as the problem list silently
# made them unselectable. Their SIF files carry their own classification
# code, which is all that is needed.
function classification_lines(dir, db)
    out = Tuple{String,Vector{SubString{String}}}[]
    seen = Set{String}()
    for line in eachline(db)
        toks = split(line)
        length(toks) >= 4 || continue
        name = String(toks[1])
        isfile(joinpath(dir, name * ".SIF")) || continue
        push!(seen, name)
        push!(out, (name, toks))
    end
    for f in sort(readdir(dir))
        endswith(f, ".SIF") || continue
        name = f[1:end-4]
        name in seen && continue
        code = nothing
        for ln in eachline(joinpath(dir, f))
            m = match(r"^\*\s+classification\s+([A-Z]{3}[0-9]-[A-Z]{2}-\S+)", ln)
            m === nothing && continue
            code = m.captures[1]
            break
        end
        code === nothing && (@warn "no classification code for $name"; continue)
        # fields 3 and 4 of the code are nvar and ncon, each a number or V,
        # which is exactly what CLASSF.DB's columns 3 and 4 carry
        parts = split(code, '-')
        length(parts) >= 4 || (@warn "malformed code for $name: $code"; continue)
        push!(out, (name, SubString.([name, code, parts[3], parts[4]], 1)))
    end
    return out
end

for (name, toks) in classification_lines(dir, db)
    cls = try
        ExaCUTEst.parse_classification(toks[2])
    catch
        global nfail += 1
        continue
    end
    try
        entries[name] = decoded_entry(name, dir, toks, cls)
    catch e
        if e isa ExaModelsSIF.ExternalFortranError
            try
                entries[name] = fortran_entry(name, toks, cls)
            catch e2
                @warn "skipping $name (external Fortran)" error = first(sprint(showerror, e2), 90)
                global nfail += 1
            end
        else
            @warn "skipping $name" error = first(sprint(showerror, e), 90)
            global nfail += 1
        end
    end
end

# Sanity check: the decode-only path must agree with a fully built model,
# which is the authority on meta. Cheap on a sample, and the only thing that
# would catch conbounds() drifting from builder.jl.
println("cross-checking decode-derived metadata against built models...")
nchecked = 0
for name in ["ROSENBR", "HS14", "HS21", "HS35", "HS45", "DEGENQP", "MATRIX2",
             "NUFFIELD", "AVGASB", "DECONVC", "GENHS28", "LUKVLE14", "CATENA"]
    haskey(entries, name) || continue
    m = ExaCUTEstModel(name).meta
    e = entries[name]
    checks = (
        ("nvar", e["variables"]["number"], m.nvar),
        ("ncon", e["constraints"]["number"], m.ncon),
        ("fixed", e["variables"]["fixed"], length(m.ifix)),
        ("free", e["variables"]["free"], length(m.ifree)),
        ("bounded_below", e["variables"]["bounded_below"], length(m.ilow)),
        ("bounded_above", e["variables"]["bounded_above"], length(m.iupp)),
        ("bounded_both", e["variables"]["bounded_both"], length(m.irng)),
        ("equality", e["constraints"]["equality"], length(m.jfix)),
        ("ineq_below", e["constraints"]["ineq_below"], length(m.jlow)),
        ("ineq_above", e["constraints"]["ineq_above"], length(m.jupp)),
        ("ineq_both", e["constraints"]["ineq_both"], length(m.jrng)),
    )
    for (field, got, want) in checks
        got == want || error("$name: decode-derived $field = $got but the " *
                             "built model says $want (conbounds/varclasses " *
                             "have drifted from builder.jl)")
    end
    global nchecked += 1
end
nchecked >= 10 || error("only $nchecked problems cross-checked; the check " *
                        "must not pass by having nothing to check")
println("  $nchecked problems agree on all 11 metadata fields")

# A partial table is worse than none: selection would silently omit problems,
# and the omission reads as "that problem is not in the collection". Earlier
# this script wrote 1537 of 1539 entries and reported the shortfall only as a
# count in a trailing line.
expected = length(classification_lines(dir, db))
length(entries) == expected || error(
    "built $(length(entries)) of $expected entries ($nfail failed): refusing " *
    "to write a partial classification table")
println("all $expected classifiable problems present")

out = joinpath(@__DIR__, "..", "src", "classf.json")
open(out, "w") do io
    JSON.print(io, entries, 2)
end
println("wrote $(length(entries)) entries to $out ($nfail skipped)")
