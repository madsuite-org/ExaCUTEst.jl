# test/oracle_assembly.jl -- roadmap step 1: verify the oracle-tier callbacks.
#
#   julia --project=. test/oracle_assembly.jl          # all 32
#   ONLY=BA-L52,HS67 julia --project=. test/oracle_assembly.jl
#
# RESULT 2026-10-16: 32/32 ASSEMBLY_OK, no mismatches
# (runs/oracle_assembly_2026-10-16.tsv).
#
# Can the 32 oracle-tier models be verified against the Fortran?
#
# NOT independently: the oracle CALLS the same compiled Fortran CUTEst.jl
# calls, so comparing the two cannot confirm the Fortran. What it CAN confirm
# is our ASSEMBLY of it -- the oracle wiring (which partials go where, the
# obj_weight/y conventions, the Lagrangian Hessian split between the scalar
# and vector oracles, the sparsity patterns). Those are ours and can be wrong
# while still "calling the Fortran", so this is the only verification
# available for this tier and it is worth having.
using ExaCUTEst, NLPModels, SparseArrays, LinearAlgebra, Random
using FileWatching.Pidfile: mkpidlock
import CUTEst
const HESS_COL_CAP = 120   # above this the per-column rebuild is too slow
log = open(get(ENV, "OUT_TSV",
    joinpath(@__DIR__, "..", "runs", "oracle_assembly.tsv")), "w")
println(log, "problem\tnvar\tncon\tverdict\tdetail")
flush(log)
rel(a, b) = begin
    m = isfinite.(a) .& isfinite.(b)
    any(m) ? maximum(abs.(a[m] .- b[m]) ./ (1 .+ max.(abs.(a[m]), abs.(b[m])))) : NaN
end

# Bounds must be compared as BOUNDS, not through rel: a vector of all-infinite
# bounds has no finite entries, so rel returns NaN, and `NaN <= 0` is false --
# which flagged AKIVA (unconstrained, all variables free) as a mismatch when
# nothing was wrong. Infinities of the same sign are equal here.
same_bounds(a, b) =
    length(a) == length(b) &&
    all(((x, y),) -> (isinf(x) && isinf(y) && sign(x) == sign(y)) || x == y, zip(a, b))
want = get(ENV, "ONLY", "")
names = isempty(want) ? ExaCUTEst.oracle_tier_problems() : split(want, ",")
for name in names
    verdict, detail = "", ""
    try
        # Build the ORACLE first and take the reference from the handle it
        # cached. Constructing our own CUTEstModel as well would be a SECOND
        # live model of the same problem, which dies with CUTEstException(1,
        # "memory allocation error") -- it made the first run of this script
        # report that for all 32, which reads like a model fault and is not
        # one. Sharing the handle is also exactly right here: the thing under
        # test is our wrapper, so the underlying Fortran model SHOULD be the
        # same object.
        # Serialize construction: CUTEst decodes and compiles every problem
        # in ONE shared directory, so concurrent builds collide. Without this
        # BA-L21LS/BA-L49/BA-L49LS reported "unable to compile a shared
        # library" and then built fine alone -- a contention artifact that
        # reads exactly like a broken problem. (BA-L52 fails either way.)
        lk = mkpidlock("/tmp/cutest-decode.pid"; wait = true)
        m = try
            ExaCUTEstModel(name)
        finally
            close(lk)
        end
        ref = ExaCUTEst._ORACLE_REFS[name]
        msgs = String[]
        m.meta.nvar == ref.meta.nvar || push!(msgs, "nvar $(m.meta.nvar) vs $(ref.meta.nvar)")
        m.meta.ncon == ref.meta.ncon || push!(msgs, "ncon $(m.meta.ncon) vs $(ref.meta.ncon)")
        same_bounds(ref.meta.lvar, m.meta.lvar) || push!(msgs, "lvar")
        same_bounds(ref.meta.uvar, m.meta.uvar) || push!(msgs, "uvar")
        ref.meta.ncon == 0 || same_bounds(ref.meta.lcon, m.meta.lcon) || push!(msgs, "lcon")
        ref.meta.ncon == 0 || same_bounds(ref.meta.ucon, m.meta.ucon) || push!(msgs, "ucon")
        Random.seed!(7)
        hess_cols = length(unique(NLPModels.hess_structure(ref)[2]))
        for (tag, x) in (("x0", copy(ref.meta.x0)),
                         ("pert", clamp.(ref.meta.x0 .+ 0.01 .* randn(ref.meta.nvar),
                                         ref.meta.lvar, ref.meta.uvar)))
            fa, fb = NLPModels.obj(ref, x), obj(m, x)
            e = abs(fa - fb) / (1 + max(abs(fa), abs(fb)))
            e <= 1e-10 || push!(msgs, "obj@$tag $(round(e, sigdigits=3))")
            e = rel(NLPModels.grad(ref, x), grad(m, x))
            e <= 1e-10 || push!(msgs, "grad@$tag $(round(e, sigdigits=3))")
            y = ref.meta.ncon > 0 ? randn(ref.meta.ncon) : Float64[]
            if ref.meta.ncon > 0
                e = rel(NLPModels.cons(ref, x), cons(m, x))
                e <= 1e-10 || push!(msgs, "cons@$tag $(round(e, sigdigits=3))")
                Ja = sparse(NLPModels.jac_structure(ref)..., NLPModels.jac_coord(ref, x),
                            ref.meta.ncon, ref.meta.nvar)
                Jb = sparse(jac_structure(m)..., jac_coord(m, x), m.meta.ncon, m.meta.nvar)
                d = maximum(abs, Ja - Jb; init = 0.0) / (1 + maximum(abs, Ja; init = 0.0))
                d <= 1e-10 || push!(msgs, "jac@$tag $(round(d, sigdigits=3))")
            end
            # Hessian via hessian-VECTOR products (Sungho 2026-10-16: "for
            # those just use hessian vectors"). One hprod per side instead of
            # materializing the matrix, which the oracle rebuilds one column
            # at a time: RAYBENDS is 30 ms this way against 63.6 s for the
            # full matrix, so every problem gets a Hessian check instead of
            # the large ones being skipped. Several random directions, since a
            # single one could miss a wrong row by landing in its null space.
            for k in 1:3
                v = randn(ref.meta.nvar)
                ha = NLPModels.hprod(ref, x, y, v)
                hb = hprod(m, x, y, v)
                e = rel(ha, hb)
                e <= 1e-9 || push!(msgs, "hprod$k@$tag $(round(e, sigdigits=3))")
            end
            if hess_cols <= HESS_COL_CAP
                Ha = sparse(NLPModels.hess_structure(ref)...,
                            NLPModels.hess_coord(ref, x, y), ref.meta.nvar, ref.meta.nvar)
                Hb = sparse(hess_structure(m)..., hess_coord(m, x, y),
                            m.meta.nvar, m.meta.nvar)
                Ha = Ha + transpose(Ha); Hb = Hb + transpose(Hb)
                d = maximum(abs, Ha - Hb; init = 0.0) / (1 + maximum(abs, Ha; init = 0.0))
                d <= 1e-9 || push!(msgs, "hess@$tag $(round(d, sigdigits=3))")
            end
        end
        hc = hess_cols <= HESS_COL_CAP ?
             "hprod x3 + full hess compared" :
             "hprod x3 compared ($hess_cols cols: full matrix not materialized)"
        verdict = isempty(msgs) ? "ASSEMBLY_OK" : "MISMATCH"
        detail = isempty(msgs) ? hc : join(msgs, " | ") * " [" * hc * "]"
        println(log, join([name, ref.meta.nvar, ref.meta.ncon, verdict, detail], '\t'))
        flush(log)
        # do NOT finalize: the oracle model still holds this handle
    catch e
        println(log, join([name, "", "", "ERROR",
                 replace(first(sprint(showerror, e), 110), '\t' => ' ', '\n' => ' ')], '\t'))
        flush(log)
    end
end
close(log)
