# Configuration comparisons (GPU against CPU, Float32 against Float64).
#
# Each row records `scale` (largest finite |reference| at that point) and
# `absdiff`; the ulp column is at-the-point, and analyze.py rescales it by the
# largest scale the same quantity reaches over all measured points.

"""
    agree(ref, test, Tw) -> NamedTuple

Normwise agreement of `test` against `ref` in units of `eps(Tw)`, `Tw` being
the lower-precision side. Non-finite entries are excluded from the difference
and reported separately as index sets.
"""
function agree(ref::AbstractVector{Float64}, test::AbstractVector{Float64}, Tw)
    length(ref) == length(test) || return (status = "LEN_DIFFER",
        n = length(ref), scale = NaN, absdiff = NaN, ulp = NaN, nfr = -1,
        nft = -1, same = "-", note = "ref $(length(ref)) test $(length(test))")
    isempty(ref) && return (status = "EMPTY", n = 0, scale = NaN,
        absdiff = NaN, ulp = NaN, nfr = 0, nft = 0, same = "yes",
        note = "zero-length reading")
    br = .!isfinite.(ref)
    bt = .!isfinite.(test)
    nfr, nft = count(br), count(bt)
    same = (nfr == 0 && nft == 0) ? "yes" : (br == bt ? "yes" : "no")
    good = .!(br .| bt)
    any(good) || return (status = "NONFINITE", n = length(ref), scale = NaN,
        absdiff = NaN, ulp = NaN, nfr = nfr, nft = nft, same = same,
        note = "every entry non-finite on one side or both")
    rg = ref[good]
    scale = maximum(abs, rg)
    absdiff = maximum(abs.(test[good] .- rg))
    # An identically zero reference cannot scale a relative error.
    if scale == 0
        return (status = absdiff == 0 ? "DEGENERATE" : "ZERO_REF",
            n = length(ref), scale = 0.0, absdiff = absdiff, ulp = NaN,
            nfr = nfr, nft = nft, same = same, note = "reference identically zero")
    end
    return (status = (nfr > 0 || nft > 0) ? "NONFINITE_PART" : "OK",
        n = length(ref), scale = scale, absdiff = absdiff,
        ulp = absdiff / (Float64(eps(Tw)) * scale), nfr = nfr, nft = nft,
        same = same, note = "")
end

"""
    agree_sparse(Ar, At, Tw)

As `agree`, for two assembled sparse matrices, read over the union of their
patterns; never densifies.
"""
function agree_sparse(Ar::SparseMatrixCSC, At::SparseMatrixCSC, Tw)
    nfr = count(!isfinite, nonzeros(Ar))
    nft = count(!isfinite, nonzeros(At))
    nzr = nonzeros(Ar)
    scale = isempty(nzr) ? 0.0 : maximum(abs, filter(isfinite, nzr); init = 0.0)
    if nfr == 0 && nft == 0
        D = Ar - At
        absdiff = isempty(nonzeros(D)) ? 0.0 : maximum(abs, nonzeros(D); init = 0.0)
        same = "yes"
    else
        dr, dt = Dict{Tuple{Int,Int},Float64}(), Dict{Tuple{Int,Int},Float64}()
        for (A, d) in ((Ar, dr), (At, dt))
            rows = rowvals(A); vals = nonzeros(A)
            for j in axes(A, 2), k in nzrange(A, j)
                d[(rows[k], j)] = vals[k]
            end
        end
        badr = Set(k for (k, v) in dr if !isfinite(v))
        badt = Set(k for (k, v) in dt if !isfinite(v))
        same = badr == badt ? "yes" : "no"
        absdiff = 0.0
        for k in union(keys(dr), keys(dt))
            (k in badr || k in badt) && continue
            absdiff = max(absdiff, abs(get(dt, k, 0.0) - get(dr, k, 0.0)))
        end
    end
    if scale == 0
        return (status = absdiff == 0 ? "DEGENERATE" : "ZERO_REF",
            n = nnz(Ar), scale = 0.0, absdiff = absdiff, ulp = NaN,
            nfr = nfr, nft = nft, same = same,
            note = "assembled reference identically zero")
    end
    return (status = (nfr > 0 || nft > 0) ? "NONFINITE_PART" : "OK",
        n = nnz(Ar), scale = scale, absdiff = absdiff,
        ulp = absdiff / (Float64(eps(Tw)) * scale), nfr = nfr, nft = nft,
        same = same, note = "")
end

# Upper triangle through `sparse`, which sums duplicates: the deduplicated and
# blockwise-with-duplicates Hessian patterns assemble to the same matrix.
asm_sym(r, c, v, n) = sparse(min.(r, c), max.(r, c), Float64.(v), n, n)
asm(r, c, v, nr, nc) = sparse(r, c, Float64.(v), nr, nc)

"""
    readings(nlp, x, y, v) -> (values, errors)

Every callback at `(x, y, v)` as host Float64 arrays, each one fault-isolated.
Structures use `similar(x, Int, n)` + the in-place `*_structure!`, which is
the form that works on a device.
"""
function readings(nlp, x, y, v)
    m = nlp.meta
    V = Dict{String,Any}()
    E = Dict{String,String}()
    f64(a) = Float64.(Array(a))
    att(k, f) = try
        V[k] = f()
    catch e
        E[k] = errsummary(e)
    end
    att("obj", () -> Float64[Float64(obj(nlp, x))])
    att("grad", () -> f64(grad(nlp, x)))
    if m.ncon > 0
        att("cons", () -> f64(cons(nlp, x)))
        att("jac_structure", function ()
            jr = similar(x, Int, m.nnzj); jc = similar(x, Int, m.nnzj)
            jac_structure!(nlp, jr, jc)
            (Array(jr), Array(jc))
        end)
        att("jac_coord", function ()
            jv = similar(x, m.nnzj); jac_coord!(nlp, x, jv); f64(jv)
        end)
    end
    att("hess_structure", function ()
        hr = similar(x, Int, m.nnzh); hc = similar(x, Int, m.nnzh)
        hess_structure!(nlp, hr, hc)
        (Array(hr), Array(hc))
    end)
    att("hess_coord", function ()
        hv = similar(x, m.nnzh)
        m.ncon > 0 ? hess_coord!(nlp, x, y, hv) : hess_coord!(nlp, x, hv)
        f64(hv)
    end)
    att("hprod", function ()
        hp = similar(x, m.nvar)
        m.ncon > 0 ? hprod!(nlp, x, y, v, hp) : hprod!(nlp, x, v, hp)
        f64(hp)
    end)
    return V, E
end

"""
    errsummary(e)

The message of `e`, led by what identifies a device failure: the kernel a GPU
compilation failed in, the `KernelError` line, and the offending argument.
"""
function errsummary(e)
    s = sprint(showerror, e)
    k = match(r"MethodInstance for ([\w.]+)\(", s)
    ke = match(r"KernelError: [^\n]*", s)
    arg = match(r"Argument \d+ to your kernel function is of type [^,\n]*", s)
    head = join(filter(!isnothing, (k === nothing ? nothing : "kernel " * k[1],
                                    ke === nothing ? nothing : ke.match,
                                    arg === nothing ? nothing : arg.match)), " | ")
    return isempty(head) ? first(s, 600) : head * " || " * first(s, 300)
end

function errstatus(cb, msg)
    if occursin("passing non-bitstype argument", msg)
        k = match(r"kernel [\w.]*\.(gpu_\w+)", msg)
        return "NONBITS_" * (k === nothing ? "?" : k[1])
    end
    if cb == "hprod"
        (occursin("Prodhelper", msg) || occursin("Scalar indexing is disallowed", msg)) &&
            return "HPROD_NO_PRODHELPER"
    end
    occursin("Out of GPU memory", msg) && return "OOM_GPU"
    return "UNAVAILABLE"
end

function emit_pair(cmp, point, Rp, Tp, Tw, nvar, ncon)
    (RV, RE), (TV, TE) = Rp, Tp
    have(cb) = if !haskey(RV, cb) || !haskey(TV, cb)
        msg = get(RE, cb, "") * (haskey(RE, cb) && haskey(TE, cb) ? " / " : "") *
              get(TE, cb, "")
        side = !haskey(RV, cb) && !haskey(TV, cb) ? "both" :
               (!haskey(RV, cb) ? "reference" : "test")
        emit(; cmp, callback = cb, point, status = errstatus(cb, msg),
             note = "$side side: $msg")
        false
    else
        true
    end
    put(cb, s) = emit(; cmp, callback = cb, point, status = s.status, n = s.n,
                      scale = s.scale, absdiff = s.absdiff, ulp = s.ulp,
                      nfin_ref = s.nfr, nfin_test = s.nft, nfin_same = s.same,
                      note = s.note)
    cm(cb) = have(cb) && (put(cb, agree(RV[cb], TV[cb], Tw)); true)
    function coords(kind, scb, vcb, acb, mk)
        if !have(scb)
            have(vcb) && cm(vcb)
            return
        end
        (rr, rc), (tr, tc) = RV[scb], TV[scb]
        okpat = (rr, rc) == (tr, tc)
        emit(; cmp, callback = scb, point, status = okpat ? "OK" : "NNZ_DIFFER",
             n = length(rr), absdiff = okpat ? 0.0 : 1.0, ulp = okpat ? 0.0 : NaN,
             nnz_ref = length(rr), nnz_test = length(tr),
             note = okpat ? "" : "$kind patterns differ")
        have(vcb) || return
        okpat && cm(vcb)
        if okpat && length(rr) > ASM_MAX
            emit(; cmp, callback = acb, point, status = "SKIPPED", n = length(rr),
                 nnz_ref = length(rr), nnz_test = length(tr),
                 note = "nnz $(length(rr)) > ASM_MAX $ASM_MAX and patterns match; " *
                        "the slot-by-slot $vcb row is equivalent")
            return
        end
        put(acb, agree_sparse(mk(rr, rc, RV[vcb]), mk(tr, tc, TV[vcb]), Tw))
    end
    cm("obj")
    cm("grad")
    if ncon > 0
        cm("cons")
        coords("jacobian", "jac_structure", "jac_coord", "jac_asm",
               (r, c, v) -> asm(r, c, v, ncon, nvar))
    end
    coords("hessian", "hess_structure", "hess_coord", "hess_asm",
           (r, c, v) -> asm_sym(r, c, v, nvar))
    cm("hprod")
    return nothing
end

"""
    points(nlp64)

`x0` and a displaced point clamped into the bounds, drawn once in Float64 and
converted per configuration; seeded by problem name, so identical on every host
and in every worker.
"""
function points(nlp64)
    x0 = Float64.(Array(nlp64.meta.x0))
    Random.seed!(hash(CUR[]) % UInt(2)^31)
    l = Float64.(Array(nlp64.meta.lvar))
    u = Float64.(Array(nlp64.meta.uvar))
    d = 0.01 .* randn(length(x0)) .* max.(1.0, abs.(x0))
    return ["x0" => x0, "xp" => clamp.(x0 .+ d, l, u)]
end

function multipliers(nlp64)
    Random.seed!(hash(CUR[]) % UInt(2)^31 + UInt(7))
    y = nlp64.meta.ncon > 0 ? randn(nlp64.meta.ncon) : Float64[]
    return y, randn(nlp64.meta.nvar)
end

# The Float64 host model's assembled Hessian times `v` against its own
# `hprod`: what says whether an `hprod` of zero is the right answer.
function selfcheck(pt, RV, v, nvar)
    (haskey(RV, "hess_structure") && haskey(RV, "hess_coord") &&
     haskey(RV, "hprod")) || return
    r, c = RV["hess_structure"]
    U = asm_sym(r, c, RV["hess_coord"], nvar)
    Hv = U * v .+ transpose(U) * v .- diag(U) .* v
    s = agree(Hv, RV["hprod"], Float64)
    emit(; cmp = "ref_selfcheck", callback = "hess_vs_hprod", point = pt,
         status = s.status, n = s.n, scale = s.scale, absdiff = s.absdiff,
         ulp = s.ulp, nfin_ref = s.nfr, nfin_test = s.nft, nfin_same = s.same,
         note = s.note)
end

function stage(f, cmp, pt)
    t0 = time()
    try
        f()
        emit(; cmp, callback = "STAGE", point = pt, status = "DONE", secs = time() - t0)
    catch e
        msg = errsummary(e)
        emit(; cmp, callback = "eval", point = pt,
             status = occursin("Out of GPU memory", msg) ? "OOM_GPU" : "ERROR",
             secs = time() - t0, note = msg)
    end
end

function build(T, backend, label)
    t0 = time()
    inflight(label * " build")
    try
        nlp = ExaCUTEstModel(CUR[]; T, backend)
        emit(; cmp = "build", callback = label, status = "BUILD_OK",
             n = nlp.meta.nvar, secs = time() - t0, nnz_ref = nlp.meta.nnzj,
             nnz_test = nlp.meta.nnzh, note = "ncon=$(nlp.meta.ncon)")
        return nlp
    catch e
        msg = errsummary(e)
        emit(; cmp = "build", callback = label,
             status = occursin("Out of GPU memory", msg) ? "OOM_GPU" : "BUILD_FAILED",
             secs = time() - t0, note = msg)
        return nothing
    end
end

"Host Float64 readings at both points: the reference for every configuration."
function host_reference(nlp64, PTS, y0, v0; self = false)
    R64 = Dict{String,Any}()
    for (pt, x) in PTS
        inflight("cpu_f64 readings $pt")
        try
            RV, RE = readings(nlp64, x, y0, v0)
            for (cb, msg) in RE
                emit(; cmp = "ref", callback = cb, point = pt,
                     status = "REF_UNAVAILABLE", note = msg)
            end
            isempty(RV) || (R64[pt] = (RV, RE))
            self && selfcheck(pt, RV, v0, nlp64.meta.nvar)
        catch e
            emit(; cmp = "ref", callback = "readings", point = pt, status = "ERROR",
                 note = first(sprint(showerror, e), 400))
        end
    end
    return R64
end
