# The BA-L* family as an ExaModel whose oracle callbacks are
# KernelAbstractions kernels over `ba_residual`, with the observation data
# resident on the model's backend.
#
# `ba_oracle_model` (BA.jl) interfaces the same `ba_residual` through host
# loops. This file's kernels let the model be built and evaluated with scalar
# indexing of device arrays disabled (`CUDA.allowscalar(false)`), which host
# loops over a device array cannot be.
#
# WHY A KERNEL AND NOT AN ExaModels EXPRESSION. Unchanged from BA.jl: the
# ~30-deep reprojection chain nests an ExaModels type parameter per operation
# and does not finish compiling. The generic function is the port; this file is
# only the launch wiring.
#
# WHAT IS PARALLEL. One observation per thread. Each observation reads exactly
# twelve variables and writes one constraint row (or one objective
# contribution), twelve Jacobian values and a 78-entry Hessian lower triangle.
# BA-L52 has 694346 observations over 192627 variables.
#
# DERIVATIVES. ForwardDiff inside the kernel, over `ba_residual`, as on the
# host: `Dual` is an immutable over an `NTuple`, so for a fixed twelve inputs
# it is stack-allocated and GPU-legal. First order uses one `Dual` level with
# twelve partials; second order nests a second level, 169 components per
# intermediate value -- heavy on registers, and it spills to local memory.
#
# SCATTER. Observations share camera and intrinsic variables, so the objective
# gradient and both Hessian-vector products scatter-accumulate with atomics.
# The constraint/Jacobian/Hessian COORDINATE callbacks do not: each observation
# owns a disjoint slice of the value array, so those kernels are pure writes
# and the duplicate (row, col) coordinates are summed by whoever assembles the
# matrix.
#
# ONE BUILDER FOR BOTH BACKENDS. `backend = nothing` runs the same kernels on
# KernelAbstractions' CPU backend. A CPU-versus-GPU comparison therefore tests
# the device rather than two unrelated implementations; `test/ba_gpu_verify.jl`
# is that comparison.
#
# Requires `hess!` on ExaModels' `ScalarNonlinearOracle` (ExaModels
# `sif-integration` 555b4ed9 or later). Without it the `*LS` variants'
# `hess_coord` has no device path: ExaModels would reconstruct the declared
# sparsity one Hessian column at a time, indexing the value array slot by slot.

using KernelAbstractions

# ---------------------------------------------------------------------------
# ForwardDiff seeding, written out rather than going through
# ForwardDiff.gradient/hessian: those take an `AbstractVector`, allocate a
# result, and consult a mutable tag counter, none of which is available in a
# kernel. Seeding by hand keeps everything in registers.

struct _BATagInner end
struct _BATagOuter end

@inline _ba_unit(::Type{T}, ::Val{N}, k) where {T, N} =
    ForwardDiff.Partials{N, T}(ntuple(j -> ifelse(j == k, one(T), zero(T)), Val(N)))

@inline _ba_zeros(::Type{T}, ::Val{N}) where {T, N} =
    ForwardDiff.Partials{N, T}(ntuple(_ -> zero(T), Val(N)))

# Outer-level seed whose entries are themselves inner Duals with no partials.
@inline function _ba_unitD(::Type{DI}, ::Type{T}, ::Val{N}, k) where {DI, T, N}
    z = _ba_zeros(T, Val(N))
    return ForwardDiff.Partials{N, DI}(
        ntuple(j -> DI(ifelse(j == k, one(T), zero(T)), z), Val(N)))
end

@inline function _ba_zerosD(::Type{DI}, ::Type{T}, ::Val{N}) where {DI, T, N}
    z = _ba_zeros(T, Val(N))
    return ForwardDiff.Partials{N, DI}(ntuple(_ -> DI(zero(T), z), Val(N)))
end

# `rodrigues` positionally: a keyword call inside a kernel is legal but makes
# the generated code harder to read.
@inline _ba_eval(v::NTuple{12, T}, y::T, rod::Bool) where {T} =
    ba_residual(v, y; rodrigues = rod)

"""
    _ba_valgrad(v, yres, rod) -> (value, gradient::NTuple{12})

One observation's projected coordinate and its gradient in the twelve
elemental variables, by one level of forward-mode AD over
[`ba_residual`](@ref).
"""
@inline function _ba_valgrad(v::NTuple{12, T}, y::T, rod::Bool) where {T}
    DI = ForwardDiff.Dual{_BATagInner, T, 12}
    d = ntuple(k -> DI(v[k], _ba_unit(T, Val(12), k)), Val(12))
    r = _ba_eval(d, DI(y, _ba_zeros(T, Val(12))), rod)
    p = ForwardDiff.partials(r)
    return ForwardDiff.value(r), ntuple(a -> p[a], Val(12))
end

"""
    _ba_valgradhess(v, yres, rod) -> (value, gradient::NTuple{12}, hessian::NTuple{12,NTuple{12}})

As [`_ba_valgrad`](@ref) plus the full 12x12 Hessian, by nesting a second
level of forward-mode AD. The Hessian comes back dense and symmetric; callers
take whichever half they need.
"""
@inline function _ba_valgradhess(v::NTuple{12, T}, y::T, rod::Bool) where {T}
    DI = ForwardDiff.Dual{_BATagInner, T, 12}
    DO = ForwardDiff.Dual{_BATagOuter, DI, 12}
    d = ntuple(
        k -> DO(DI(v[k], _ba_unit(T, Val(12), k)), _ba_unitD(DI, T, Val(12), k)),
        Val(12))
    yd = DO(DI(y, _ba_zeros(T, Val(12))), _ba_zerosD(DI, T, Val(12)))
    r = _ba_eval(d, yd, rod)
    ri = ForwardDiff.value(r)           # inner Dual: value + gradient
    op = ForwardDiff.partials(r)        # Partials{12,DI}: one gradient per direction
    gi = ForwardDiff.partials(ri)
    return (ForwardDiff.value(ri),
            ntuple(a -> gi[a], Val(12)),
            ntuple(a -> (pa = ForwardDiff.partials(op[a]);
                         ntuple(b -> pa[b], Val(12))), Val(12)))
end

"The 78 lower-triangle slots of a 12x12 block, in (row, col) order."
const _BA_LT = Tuple((a, b) for a in 1:12 for b in 1:a)

# Tuple sum and dot by inlined recursion, NOT by a `sum` over a closure that
# updates a captured accumulator: capturing a reassigned variable boxes it on
# the heap, and a heap allocation inside a kernel is an `InvalidIRError`, not a
# slowdown.
@inline _ba_tsum(t::Tuple{Any}) = t[1]
@inline _ba_tsum(t::Tuple) = t[1] + _ba_tsum(Base.tail(t))
@inline _ba_dot(a::NTuple{N, T}, b::NTuple{N, T}) where {N, T} =
    _ba_tsum(ntuple(k -> a[k] * b[k], Val(N)))

# ---------------------------------------------------------------------------
# Kernels. One observation per thread; `idx` is 12 x nobs, so a thread's twelve
# variable indices are contiguous in the first dimension. Every `idx` entry is
# in `1:length(x)` -- `_ba_check_indices` establishes that at build time.

@kernel function _ba_cons_kernel!(
        cv, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(bc), rod)
    i = @index(Global)
    v = ntuple(k -> x[idx[k, i]], Val(12))
    cv[i] = w[i] * _ba_eval(v, yres[i], rod) - bc[i]
end

@kernel function _ba_jac_kernel!(
        vals, @Const(x), @Const(idx), @Const(yres), @Const(w), rod)
    i = @index(Global)
    v = ntuple(k -> x[idx[k, i]], Val(12))
    _, g = _ba_valgrad(v, yres[i], rod)
    wi = w[i]
    base = (i - 1) * 12
    ntuple(a -> (vals[base + a] = wi * g[a]; nothing), Val(12))
end

# Constraint-block Hessian: row i contributes y[i] * w[i] * nabla^2 ba.
@kernel function _ba_conhess_kernel!(
        vals, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(mult), rod)
    i = @index(Global)
    v = ntuple(k -> x[idx[k, i]], Val(12))
    _, _, H = _ba_valgradhess(v, yres[i], rod)
    s = w[i] * mult[i]
    base = (i - 1) * 78
    ntuple(Val(78)) do p
        a, b = _BA_LT[p]
        vals[base + p] = s * H[a][b]
        nothing
    end
end

# Constraint-block Hessian-vector product, (sum_i y_i nabla^2 c_i) * v.
@kernel function _ba_conhvp_kernel!(
        Hv, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(mult),
        @Const(vv), rod)
    i = @index(Global)
    jj = ntuple(k -> idx[k, i], Val(12))
    v = ntuple(k -> x[jj[k]], Val(12))
    _, _, H = _ba_valgradhess(v, yres[i], rod)
    s = w[i] * mult[i]
    vl = ntuple(k -> vv[jj[k]], Val(12))
    ntuple(Val(12)) do a
        KernelAbstractions.@atomic Hv[jj[a]] += s * _ba_dot(H[a], vl)
        nothing
    end
end

# --- least-squares (objective) variants ---

@kernel function _ba_sq_kernel!(
        buf, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(bc), rod)
    i = @index(Global)
    v = ntuple(k -> x[idx[k, i]], Val(12))
    r = w[i] * _ba_eval(v, yres[i], rod) - bc[i]
    buf[i] = r * r
end

@kernel function _ba_objgrad_kernel!(
        g, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(bc), rod)
    i = @index(Global)
    jj = ntuple(k -> idx[k, i], Val(12))
    v = ntuple(k -> x[jj[k]], Val(12))
    val, gr = _ba_valgrad(v, yres[i], rod)
    wi = w[i]
    r = wi * val - bc[i]
    c = 2 * r * wi
    ntuple(a -> (KernelAbstractions.@atomic g[jj[a]] += c * gr[a]; nothing), Val(12))
end

# Exact Hessian of sum_i r_i^2: 2 (J'J + sum_i r_i nabla^2 r_i), one 12x12
# block per observation, lower triangle.
@kernel function _ba_objhess_kernel!(
        vals, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(bc), rod)
    i = @index(Global)
    v = ntuple(k -> x[idx[k, i]], Val(12))
    val, gr, H = _ba_valgradhess(v, yres[i], rod)
    wi = w[i]
    r = wi * val - bc[i]
    base = (i - 1) * 78
    ntuple(Val(78)) do p
        a, b = _BA_LT[p]
        vals[base + p] = 2 * (wi * wi * gr[a] * gr[b] + r * wi * H[a][b])
        nothing
    end
end

@kernel function _ba_objhvp_kernel!(
        Hv, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(bc),
        @Const(vv), rod)
    i = @index(Global)
    jj = ntuple(k -> idx[k, i], Val(12))
    v = ntuple(k -> x[jj[k]], Val(12))
    val, gr, H = _ba_valgradhess(v, yres[i], rod)
    wi = w[i]
    r = wi * val - bc[i]
    vl = ntuple(k -> vv[jj[k]], Val(12))
    gv = _ba_dot(gr, vl)
    ntuple(Val(12)) do a
        KernelAbstractions.@atomic Hv[jj[a]] +=
            2 * wi * (wi * gr[a] * gv + r * _ba_dot(H[a], vl))
        nothing
    end
end

# ---------------------------------------------------------------------------
# Device-resident observation data.

"""
    BAKernelData

One bundle-adjustment problem's observations as a struct of arrays on whatever
device the model is built for: `idx` is 12 x nobs variable indices, `yres`
selects the pixel component, `w` is the group weight and `bc` the observed
coordinate (the SIF group constant).
"""
struct BAKernelData{MI, VT}
    idx::MI
    yres::VT
    w::VT
    bc::VT
    nobs::Int
end

"""
    _ba_check_indices(obs, nvar)

Check that every variable index of every observation lands in `1:nvar`.

The kernels gather `x[idx[k, i]]` with no bounds check of their own, so this is
what makes that legal. It runs once per build, O(nobs), against callbacks that
cost milliseconds. On a device an out-of-range gather has no useful diagnosis:
it reads whatever memory is there, or aborts the context, with nothing
pointing back at the decode that produced the index.
"""
function _ba_check_indices(obs, nvar::Integer)
    for (i, o) in pairs(obs)
        for k in 1:12
            j = o.idx[k]
            1 <= j <= nvar || error(
                "observation $i maps elemental variable $k to index $j, " *
                "outside 1:$nvar")
        end
    end
    return nothing
end

function _ba_kernel_data(obs, ::Type{T}, backend) where {T}
    nobs = length(obs)
    idxh = Matrix{Int32}(undef, 12, nobs)
    yresh = Vector{T}(undef, nobs)
    wh = Vector{T}(undef, nobs)
    bh = Vector{T}(undef, nobs)
    for (i, o) in pairs(obs)
        for k in 1:12
            idxh[k, i] = o.idx[k]
        end
        yresh[i] = o.yres
        wh[i] = o.w
        bh[i] = o.b
    end
    cv = v -> ExaModels.convert_array(v, backend)
    return BAKernelData(cv(idxh), cv(yresh), cv(wh), cv(bh), nobs)
end

# ---------------------------------------------------------------------------

"""
    ba_kernel_model(name; T = Float64, backend = nothing, rodrigues = false) -> ExaModel

Build a BA-L* problem as an ExaModel whose oracle callbacks are
KernelAbstractions kernels over [`ba_residual`](@ref), with the observation
data resident on `backend`.

`backend = nothing` builds a host model that runs the same kernels on
KernelAbstractions' CPU backend; `backend = CUDABackend()` builds a device
model that never indexes a device array from the host, so it is usable with
`CUDA.allowscalar(false)`.

The constrained variants (`BA-L1`, `BA-L16`, ...) become a
`VectorNonlinearOracle` with one row per observation; the `*LS` variants, whose
objective is the sum of squared residuals and which have no constraints,
become a `ScalarNonlinearOracle`.
"""
function ba_kernel_model(name::AbstractString; T = Float64, backend = nothing,
                         rodrigues::Bool = false)
    P = ExaModelsSIF.sif_decode(sif_problem_path(name); external = :defer)
    obs, _rows, isobj = ba_observations(P, T)
    n = length(P.varnames)
    nobs = length(obs)
    _ba_check_indices(obs, n)
    d = _ba_kernel_data(obs, T, backend)
    ka = backend === nothing ? KernelAbstractions.CPU() : backend
    rod = rodrigues

    c = ExaModels.ExaCore(T; backend = backend)
    c, _x = ExaModels.add_var(c, n; start = P.x0, lvar = P.lvar, uvar = P.uvar)

    # Hessian sparsity: the lower triangle of each observation's 12x12 block,
    # in the slot order the kernels write. Duplicate (i, j) across observations
    # are intended -- ExaModels sums coordinate duplicates.
    hr = Vector{Int}(undef, 78 * nobs)
    hc = Vector{Int}(undef, 78 * nobs)
    let p = 0
        for o in obs, a in 1:12, b in 1:a
            p += 1
            i, j = o.idx[a], o.idx[b]
            hr[p], hc[p] = max(i, j), min(i, j)
        end
    end

    if isobj
        sqbuf = ExaModels.convert_array(zeros(T, nobs), backend)
        f = xv -> begin
            _ba_sq_kernel!(ka)(sqbuf, xv, d.idx, d.yres, d.w, d.bc, rod;
                               ndrange = nobs)
            KernelAbstractions.synchronize(ka)
            return sum(sqbuf)
        end
        grad! = (g, xv) -> begin
            fill!(g, zero(eltype(g)))
            _ba_objgrad_kernel!(ka)(g, xv, d.idx, d.yres, d.w, d.bc, rod;
                                    ndrange = nobs)
            KernelAbstractions.synchronize(ka)
            return g
        end
        hess! = (vals, xv) -> begin
            _ba_objhess_kernel!(ka)(vals, xv, d.idx, d.yres, d.w, d.bc, rod;
                                    ndrange = nobs)
            KernelAbstractions.synchronize(ka)
            return vals
        end
        hvp! = (Hv, xv, vv) -> begin
            fill!(Hv, zero(eltype(Hv)))
            _ba_objhvp_kernel!(ka)(Hv, xv, d.idx, d.yres, d.w, d.bc, vv, rod;
                                   ndrange = nobs)
            KernelAbstractions.synchronize(ka)
            return Hv
        end
        sor = ExaModels.ScalarNonlinearOracle(;
            nvar = n, f = f, grad! = grad!, hvp! = hvp!, hess! = hess!,
            hess_rows = hr, hess_cols = hc, adapt = Val(false))
        c = ExaModels.objective(c, sor)
    else
        jr = Vector{Int}(undef, 12 * nobs)
        jc = Vector{Int}(undef, 12 * nobs)
        let p = 0
            for (i, o) in pairs(obs), a in 1:12
                p += 1
                jr[p], jc[p] = i, o.idx[a]
            end
        end
        vor = ExaModels.VectorNonlinearOracle(;
            nvar = n, ncon = nobs,
            jac_rows = jr, jac_cols = jc, hess_rows = hr, hess_cols = hc,
            lcon = zeros(T, nobs), ucon = zeros(T, nobs),
            f! = (cv, xv) -> begin
                _ba_cons_kernel!(ka)(cv, xv, d.idx, d.yres, d.w, d.bc, rod;
                                     ndrange = nobs)
                KernelAbstractions.synchronize(ka)
                return cv
            end,
            jac! = (vals, xv) -> begin
                _ba_jac_kernel!(ka)(vals, xv, d.idx, d.yres, d.w, rod;
                                    ndrange = nobs)
                KernelAbstractions.synchronize(ka)
                return vals
            end,
            hess! = (vals, xv, yv) -> begin
                _ba_conhess_kernel!(ka)(vals, xv, d.idx, d.yres, d.w, yv, rod;
                                        ndrange = nobs)
                KernelAbstractions.synchronize(ka)
                return vals
            end,
            hvp! = (Hv, xv, yv, vv) -> begin
                fill!(Hv, zero(eltype(Hv)))
                _ba_conhvp_kernel!(ka)(Hv, xv, d.idx, d.yres, d.w, yv, vv, rod;
                                       ndrange = nobs)
                KernelAbstractions.synchronize(ka)
                return Hv
            end,
            adapt = Val(false))
        c = ExaModels.constraint(c, vor)
    end
    return ExaModels.ExaModel(c)
end
