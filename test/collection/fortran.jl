# Part A: CPU Float64 against the compiled Fortran (CUTEst.jl).
#
# Decoded problems follow ExaModelsSIF's test/pilot.jl (its tolerance, its
# 1+max relative error, and its reference-defect lists HESS_REF_BUGS,
# DERIV_REF_BUGS, VALUE_REF_BUGS, REF_DECODE_BUGS, REF_NOISE, REF_NOISE_TOL),
# loaded from the ExaModelsSIF this environment resolves. Two differences from
# pilot.jl's `check`:
# the model is reached through `ExaCUTEstModel(name)`, and `hprod` is compared
# as well, under the same Hessian adjudication as `hess` (the reference's hprod
# is built from the same hand-coded H lines).
#
# Native problems run `verify` from this repository's test/native_verify.jl,
# with its FIRST_MODE / HESS_MODE / TOL tables, loaded with two substitutions,
# each asserted to occur exactly once: its driver section is dropped, and the
# host-local pidlock that wrapped the whole comparison becomes the run's shared
# lock around CUTEstModel construction only.

using ExaModelsSIF
include(joinpath(pkgdir(ExaModelsSIF), "test", "pilot.jl"))

let path = joinpath(@__DIR__, "..", "native_verify.jl")
    src = read(path, String)
    k = findfirst("# --- driver ---", src)
    k === nothing && error("native_verify.jl: driver marker not found")
    src = src[1:first(k)-1]
    for (a, b) in ("mkpidlock(LOCK) do" => "(f -> f())() do",
                   "ref = CUTEstModel(name)" => "ref = with_cutest_lock(() -> CUTEstModel(name))")
        count(a, src) == 1 || error("native_verify.jl: expected one `$a`")
        src = replace(src, a => b)
    end
    include_string(Main, src, path)
end

function emitA(prob; callback, point = "-", status, val = NaN, tol = NaN,
               absdiff = NaN, scale = NaN, mode = "", note = "")
    row = join((prob, tier(), "cpu64_vs_fortran", callback, point, status, val, tol,
                absdiff, scale, clean(mode), HOST, WID, clean(note)), '\t')
    append_row(TSV_A, row)
end

# abs difference and reference scale over entries where the reference is finite
function absscale(a, b)
    m = isfinite.(a)
    any(m) || return (NaN, NaN)
    return maximum(abs.(a[m] .- b[m])), maximum(abs, a[m])
end
function absscale(A::SparseMatrixCSC, B::SparseMatrixCSC)
    D = A - B
    v = filter(isfinite, nonzeros(D))
    s = filter(isfinite, nonzeros(A))
    return (isempty(v) ? 0.0 : maximum(abs, v)), (isempty(s) ? 0.0 : maximum(abs, s))
end

function fortran_native(name)
    checks, tb, nv, nc = verify(name)
    for c in checks
        note = isempty(c.mode) ? why(name, c.cb) : c.mode
        emitA(name; callback = c.cb, point = c.pt, status = ok(c) ? "ok" : "BAD",
              val = c.val, tol = c.tol, mode = "native_verify", note)
    end
    emitA(name; callback = "PROBLEM", status = all(ok, checks) ? "PASS" : "FAIL",
          val = length(checks), note = "nvar=$nv ncon=$nc build_s=$(round(tb, digits=1))")
end

function fortran_decoded(name, m)
    tol = name in REF_NOISE ? get(REF_NOISE_TOL, name, 1e-4) : 1e-6
    tolnote = name in REF_NOISE ? "REF_NOISE" : ""
    nbad = Ref(0); nchk = Ref(0)
    rec(cb, pt, e, a, s; adjud = false, why = "") = begin
        nchk[] += 1
        st = isnan(e) ? "BAD" : e <= tol ? "ok" : adjud ? "REF_ADJUDICATED" : "BAD"
        st == "BAD" && (nbad[] += 1)
        emitA(name; callback = cb, point = pt, status = st, val = e, tol, absdiff = a,
              scale = s, mode = tolnote, note = st == "REF_ADJUDICATED" ? why : "")
    end
    inflight("fortran build")
    ref = with_cutest_lock(() -> CUTEstModel(name))
    try
        inflight("fortran compare")
        nv, nc = ref.meta.nvar, ref.meta.ncon
        if nv != m.meta.nvar || nc != m.meta.ncon
            emitA(name; callback = "dims", status = "BAD",
                  note = "ref $nv/$nc ours $(m.meta.nvar)/$(m.meta.ncon)")
            nbad[] += 1
            return
        end
        ovn = ExaCUTEst.sif_variable_names(name)
        ocn = nc > 0 ? ExaCUTEst.sif_constraint_names(name) : String[]
        # By Dict: a findfirst per name is quadratic in the name count.
        ivn = Dict(s => j for (j, s) in enumerate(ovn))
        icn = Dict(s => i for (i, s) in enumerate(ocn))
        vp = Int[ivn[s] for s in refvarnames(ref)]      # our index of ref var k
        perm = nc > 0 ? Int[icn[s] for s in refconnames(ref)] : Int[]
        (length(ivn) == nv && sort(vp) == 1:nv) || error("variable map not a bijection")
        (nc == 0 || (length(icn) == nc && sort(perm) == 1:nc)) ||
            error("row map not a bijection")
        for (fld, a, b) in (("x0", ref.meta.x0, m.meta.x0[vp]),
                            ("lvar", ref.meta.lvar, m.meta.lvar[vp]),
                            ("uvar", ref.meta.uvar, m.meta.uvar[vp]),
                            ("lcon", ref.meta.lcon, m.meta.lcon[perm]),
                            ("ucon", ref.meta.ucon, m.meta.ucon[perm]))
            nc == 0 && fld in ("lcon", "ucon") && continue
            a2 = replace(a, Inf => 1e30, -Inf => -1e30)
            b2 = replace(b, Inf => 1e30, -Inf => -1e30)
            rec(fld, "-", relerr(a2, b2), absscale(a2, b2)...)
        end
        Random.seed!(7)
        pts = [("x0", copy(ref.meta.x0))]
        push!(pts, ("pert", clamp.(ref.meta.x0 .+ 0.01 .* randn(nv),
                                   ref.meta.lvar, ref.meta.uvar)))
        for (set, why) in ((VALUE_REF_BUGS, "VALUE_REF_BUGS: reference value function contradicts the SIF"),
                           (REF_DECODE_BUGS, "REF_DECODE_BUGS: reference decode wrong"))
            if name in set
                emitA(name; callback = "values", status = "REF_ADJUDICATED", note = why)
                return
            end
        end
        dgate = name in DERIV_REF_BUGS
        hgate = name in HESS_REF_BUGS || dgate
        dwhy = "DERIV_REF_BUGS: reference G/H are not the derivative of its F"
        hwhy = dgate ? dwhy : "HESS_REF_BUGS: hand-coded reference H contradicts fd of the reference gradient"
        npts = 0
        for (tag, xp) in pts
            xm = zeros(nv); xm[vp] = xp
            fa = obj(ref, xp); fb = obj(m, xm)
            if !isfinite(fa)
                emitA(name; callback = "point", point = tag, status = "REF_NONFINITE",
                      note = "reference objective $fa at this point")
                continue
            end
            npts += 1
            rec("obj", tag, abs(fa - fb) / (1.0 + max(abs(fa), abs(fb))), abs(fa - fb), abs(fa))
            gr = grad(ref, xp); go = grad(m, xm)[vp]
            rec("grad", tag, relerr(gr, go), absscale(gr, go)...; adjud = dgate, why = dwhy)
            if nc > 0
                cr = cons(ref, xp); co = cons(m, xm)[perm]
                rec("cons", tag, relerr(cr, co), absscale(cr, co)...)
                Ja = spjac(ref, xp); Jb = spjac(m, xm)[perm, vp]
                rec("jac", tag, sprelerr(Ja, Jb), absscale(Ja, Jb)...; adjud = dgate, why = dwhy)
            end
            y = nc > 0 ? randn(nc) : Float64[]
            ym = zeros(nc); ym[perm] = y
            Ha = sphess(ref, xp, y)
            Hbf = sphess(m, xm, ym)
            Hbp = (Hbf + transpose(Hbf))[vp, vp]
            Hap = Ha + transpose(Ha)
            rec("hess", tag, sprelerr(Hap, Hbp), absscale(Hap, Hbp)...; adjud = hgate, why = hwhy)
            Ha = Hbf = Hbp = Hap = nothing
            v = randn(nv); vm = zeros(nv); vm[vp] = v
            hr = nc > 0 ? hprod(ref, xp, y, v) : hprod(ref, xp, v)
            ho = (nc > 0 ? hprod(m, xm, ym, vm) : hprod(m, xm, vm))[vp]
            rec("hprod", tag, relerr(hr, ho), absscale(hr, ho)...; adjud = hgate, why = hwhy)
        end
        if npts == 0
            emitA(name; callback = "point", status = "BAD",
                  note = "reference NaN at every test point (domain issue)")
            nbad[] += 1
        end
    catch e
        emitA(name; callback = "error", status = "BAD", note = first(sprint(showerror, e), 400))
        nbad[] += 1
    finally
        finalize(ref)
        emitA(name; callback = "PROBLEM", status = nbad[] == 0 ? "PASS" : "FAIL",
              val = nchk[], note = "nvar=$(m.meta.nvar) ncon=$(m.meta.ncon)")
    end
end
