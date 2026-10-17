# BA_gpu.jl -- the BA-L* family on a GPU (roadmap step 4).
#
# src/native/BA.jl ports the bundle-adjustment reprojection to a generic Julia
# function, `ba_residual`, and wires it to ExaModels as an external oracle
# whose callbacks are host loops. This file replaces those loops with
# KernelAbstractions kernels over the observations, so the SAME `ba_residual`
# runs on the device and the model is constructible and evaluable with
# `CUDA.allowscalar(false)` in force.
#
# WHY A KERNEL AND NOT AN ExaModels EXPRESSION. Unchanged from BA.jl: built as
# a symbolic expression the ~30-deep reprojection chain did not finish
# compiling in an hour on a 57-variable problem. The generic function is the
# port; this file is only the launch wiring.
#
# WHAT IS PARALLEL. One observation per thread. Each observation reads exactly
# 12 variables, writes 1 constraint row (or one objective contribution), 12
# Jacobian values and a 78-entry Hessian lower triangle. BA-L52 has 694346
# observations over 192627 variables, which is as embarrassingly parallel as
# this collection gets.
#
# DERIVATIVES. ForwardDiff inside the kernel, over `ba_residual`, exactly as on
# the host: `Dual` is a plain immutable over an `NTuple`, so for a fixed 12
# inputs it is stack-allocated and GPU-legal. First order uses one `Dual`
# level with 12 partials; second order nests a second level, 169 components per
# intermediate value. That is heavy on registers and spills to local memory,
# but it compiles and runs -- measured, not assumed (see
# ~/git/scratch/ba_gpu/verify_ba_gpu.jl).
#
# SCATTER. Observations share camera and intrinsic variables, so the objective
# gradient and the Hessian-vector products scatter-accumulate with atomics.
# The constraint/Jacobian/Hessian *coordinate* callbacks do not: each
# observation owns a disjoint slice of the value array, so those kernels are
# pure writes and the duplicate (row, col) coordinates are summed later by
# whoever assembles the matrix.
#
# ONE BUILDER FOR BOTH SIDES. `backend = nothing` runs the same kernels on
# KernelAbstractions' CPU backend rather than taking a different code path, so
# a CPU-versus-GPU comparison tests the device, not two unrelated
# implementations. `ba_oracle_model` (BA.jl) remains the host reference that
# was verified against the Fortran.

using KernelAbstractions

# ---------------------------------------------------------------------------
# ForwardDiff seeding, written out rather than going through
# ForwardDiff.gradient/hessian: those take an `AbstractVector`, allocate a
# result, and consult a mutable tag counter -- none of which is available in a
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

# `rodrigues` as a positional `Bool` -- a keyword call inside a kernel is
# legal but needlessly hard to read in the generated code.
@inline _ba_eval(v::NTuple{12, T}, y::T, rod::Bool) where {T} =
    ba_residual(v, y; rodrigues = rod)

"""
    _ba_valgrad(v, yres, rod) -> (value, gradient::NTuple{12})

One observation's projected coordinate and its gradient in the 12 elemental
variables, by one level of forward-mode AD over [`ba_residual`](@ref).
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

# ---------------------------------------------------------------------------
# Kernels. One observation per thread; `idx` is 12 x nobs so a thread's twelve
# variable indices are contiguous in the first dimension.

@kernel function _ba_cons_kernel!(
        cv, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(bc), rod)
    i = @index(Global)
    @inbounds begin
        v = ntuple(k -> x[idx[k, i]], Val(12))
        cv[i] = w[i] * _ba_eval(v, yres[i], rod) - bc[i]
    end
end

@kernel function _ba_jac_kernel!(
        vals, @Const(x), @Const(idx), @Const(yres), @Const(w), rod)
    i = @index(Global)
    @inbounds begin
        v = ntuple(k -> x[idx[k, i]], Val(12))
        _, g = _ba_valgrad(v, yres[i], rod)
        wi = w[i]
        base = (i - 1) * 12
        ntuple(a -> (vals[base + a] = wi * g[a]; nothing), Val(12))
    end
end

# Constraint-block Hessian: row i contributes y[i] * w[i] * nabla^2 ba.
@kernel function _ba_conhess_kernel!(
        vals, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(mult), rod)
    i = @index(Global)
    @inbounds begin
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
end

# Constraint-block Hessian-vector product, (sum_i y_i nabla^2 c_i) * v.
@kernel function _ba_conhvp_kernel!(
        Hv, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(mult),
        @Const(vv), rod)
    i = @index(Global)
    @inbounds begin
        jj = ntuple(k -> idx[k, i], Val(12))
        v = ntuple(k -> x[jj[k]], Val(12))
        _, _, H = _ba_valgradhess(v, yres[i], rod)
        s = w[i] * mult[i]
        vl = ntuple(k -> vv[jj[k]], Val(12))
        ntuple(Val(12)) do a
            acc = zero(eltype(Hv))
            ntuple(b -> (acc += H[a][b] * vl[b]; nothing), Val(12))
            KernelAbstractions.@atomic Hv[jj[a]] += s * acc
            nothing
        end
    end
end

# --- least-squares (objective) variants ---

@kernel function _ba_sq_kernel!(
        buf, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(bc), rod)
    i = @index(Global)
    @inbounds begin
        v = ntuple(k -> x[idx[k, i]], Val(12))
        r = w[i] * _ba_eval(v, yres[i], rod) - bc[i]
        buf[i] = r * r
    end
end

@kernel function _ba_objgrad_kernel!(
        g, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(bc), rod)
    i = @index(Global)
    @inbounds begin
        jj = ntuple(k -> idx[k, i], Val(12))
        v = ntuple(k -> x[jj[k]], Val(12))
        val, gr = _ba_valgrad(v, yres[i], rod)
        wi = w[i]
        r = wi * val - bc[i]
        c = 2 * r * wi
        ntuple(a -> (KernelAbstractions.@atomic g[jj[a]] += c * gr[a]; nothing), Val(12))
    end
end

# Exact Hessian of sum_i r_i^2: 2 (J'J + sum_i r_i nabla^2 r_i), one 12x12
# block per observation, lower triangle.
@kernel function _ba_objhess_kernel!(
        vals, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(bc), rod)
    i = @index(Global)
    @inbounds begin
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
end

@kernel function _ba_objhvp_kernel!(
        Hv, @Const(x), @Const(idx), @Const(yres), @Const(w), @Const(bc),
        @Const(vv), rod)
    i = @index(Global)
    @inbounds begin
        jj = ntuple(k -> idx[k, i], Val(12))
        v = ntuple(k -> x[jj[k]], Val(12))
        val, gr, H = _ba_valgradhess(v, yres[i], rod)
        wi = w[i]
        r = wi * val - bc[i]
        vl = ntuple(k -> vv[jj[k]], Val(12))
        gv = zero(eltype(Hv))
        ntuple(a -> (gv += gr[a] * vl[a]; nothing), Val(12))
        ntuple(Val(12)) do a
            acc = 2 * wi * wi * gr[a] * gv
            ntuple(b -> (acc += 2 * r * wi * H[a][b] * vl[b]; nothing), Val(12))
            KernelAbstractions.@atomic Hv[jj[a]] += acc
            nothing
        end
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

function _ba_kernel_data(obs, ::Type{T}, backend) where {T}
    nobs = length(obs)
    idxh = Matrix{Int32}(undef, 12, nobs)
    yresh = Vector{T}(undef, nobs)
    wh = Vector{T}(undef, nobs)
    bh = Vector{T}(undef, nobs)
    @inbounds for (i, o) in enumerate(obs)
        for k in 1:12
            idxh[k, i] = o.idx[k]
        end
        yresh[i] = T(o.yres)
        wh[i] = T(o.w)
        bh[i] = T(o.b)
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
    obs, _rows, isobj = ba_observations(P)
    n = length(P.varnames)
    nobs = length(obs)
    d = _ba_kernel_data(obs, T, backend)
    ka = backend === nothing ? KernelAbstractions.CPU() : backend
    rod = rodrigues

    c = ExaModels.ExaCore(T; backend = backend)
    c, _x = ExaModels.add_var(c, n; start = P.x0, lvar = P.lvar, uvar = P.uvar)

    # Hessian sparsity: the lower triangle of each observation's 12x12 block,
    # in the same slot order the kernels write. Duplicate (i, j) across
    # observations are intended -- ExaModels sums coordinate duplicates.
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
            for (i, o) in enumerate(obs), a in 1:12
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
