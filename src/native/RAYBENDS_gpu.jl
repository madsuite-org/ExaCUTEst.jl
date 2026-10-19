# RAYBENDS and BLEACHNG as ExaModels whose objective callbacks are
# KernelAbstractions kernels over `raybends_time` and `bleachng_conc`, with the
# per-item data resident on the model's backend.
#
# `raybends_model` / `bleachng_model` (RAYBENDS.jl) interface the same two
# functions through host loops. This file's kernels let the models be built and
# evaluated with scalar indexing of device arrays disabled
# (`CUDA.allowscalar(false)`), which host loops over a device array cannot be.
# Both pairs of builders share `_raybends_decode` / `_bleachng_decode`, so the
# SIF reading has one definition.
#
# The two problems share this file for the same reason they share RAYBENDS.jl:
# one external element type per problem, objective only, the value a plain
# generic function of a handful of elemental variables, and every derivative
# ForwardDiff over that function.
#
# WHAT IS PARALLEL.
#
#   * RAYBENDS: one spline segment per thread, 1026 of them over 2050
#     variables, each reading the 8 elemental variables of its four knots. The
#     4x11 basis table is pure constants, built once per model
#     (`raybends_table`) and passed to every kernel by value.
#   * BLEACHNG: one observation per thread, 1602 of them over 17 variables.
#     That is all the parallelism the problem has -- there is no inner
#     integration to spread out, because the port solves the ODE in closed
#     form -- so `ndrange` is 1602 and nothing is invented beyond it. The 1602
#     purely linear groups stay an ordinary SIMD ExaModels objective, exactly
#     as in `bleachng_model`.
#
# The Hessian runs one thread per (item, column) instead, as in
# `JIMACK_gpu.jl`: a column of an item's block is one directional derivative of
# the item gradient, the columns are independent, and it multiplies the thread
# count by 8 and by 5 respectively.
#
# DERIVATIVES. The kernels call `raybends_grad`, `raybends_hvp`,
# `bleachng_grad` and `bleachng_grad_hvp` from RAYBENDS.jl unchanged: those
# seed ForwardDiff by hand over tuples, so there is no config to allocate and
# no mutable tag counter to consult, and the duals stay in registers. A Hessian
# column is the same `hvp` with `e_j` for the direction.
#
# NO ATOMICS ANYWHERE, so every callback is bitwise reproducible across
# backends and both builders serve a non-bits element type.
#
# Items overlap -- RAYBENDS' segments share knots (and `T(0)` repeats one knot
# three times within a single element), BLEACHNG's observations all share
# `kLa` -- so the gradient, the Hessian-vector product and the Hessian all
# need accumulation. Each runs as TWO kernels rather than one atomic one:
#
#   1. one thread per item (or per (item, column)) writes that item's own
#      contributions into its own slice of a scratch array -- a disjoint write;
#   2. one thread per OUTPUT slot sums the scratch entries listed for it in a
#      table built once at model build time (`_rbg_gather_table`).
#
# The summation order is that table's, so it is the same on every backend and
# in every run. The cost is one extra pass over `8 x nel` or `36 x nel` values
# against a kernel launch that would otherwise serialize on contended
# addresses -- every one of BLEACHNG's 1602 observations writes `kLa`.
#
# HESSIAN PATTERN: THE DEDUPLICATED UNION, which is what `raybends_model` and
# `bleachng_model` declare and, for RAYBENDS, exactly the 15351 nonzeros CUTEst
# reports. `ExaModels.hess_coord!` writes one value per declared entry and the
# assembler sums duplicates, so with a deduplicated pattern `hess!` must
# deliver each entry ALREADY ACCUMULATED -- which is what stage 2 above does,
# since an output slot there is a declared Hessian entry rather than a
# variable. The blockwise-with-duplicates alternative (`JIMACK_gpu.jl`,
# `COLLOC_gpu.jl`) would make stage 2 unnecessary, but here it costs nothing:
# the gather is the same machinery the gradient already needs, and matching the
# host builders' pattern entry for entry lets the two be compared slot by slot
# rather than only as assembled matrices. For BLEACHNG it also matters for its
# own sake: 15 slots x 1602 observations is 24030 declared entries over a
# 17-variable problem whose deduplicated union is 57.
#
# Requires `hess!` on ExaModels' `ScalarNonlinearOracle` (ExaModels
# `sif-integration` 555b4ed9 or later). Without it `hess_coord` has no device
# path at all: ExaModels reconstructs the declared pattern one Hessian column
# at a time, which for RAYBENDS is 2050 passes over all 1026 segments.
#
# ONE BUILDER FOR BOTH BACKENDS. `backend = nothing` runs the same kernels on
# KernelAbstractions' CPU backend. A CPU-versus-GPU comparison therefore tests
# the device rather than two unrelated implementations;
# `test/raybends_gpu_verify.jl` is that comparison.

using KernelAbstractions

# ---------------------------------------------------------------------------
# Tuple reductions by inlined recursion, NOT by a `sum` over a closure that
# updates a captured accumulator: capturing a reassigned variable boxes it on
# the heap, and a heap allocation inside a kernel is an `InvalidIRError`, not a
# slowdown.

@inline _rbg_tsum(t::Tuple{Any}) = t[1]
@inline _rbg_tsum(t::Tuple) = t[1] + _rbg_tsum(Base.tail(t))

@inline _rbg_tdot(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T} =
    _rbg_tsum(ntuple(k -> a[k] * b[k], Val(N)))

# ---------------------------------------------------------------------------
# The scatter-as-gather table, shared by both problems in this file.

"""
    _rbg_gather_table(srcpos, dest, ndest) -> (ptr, src)
    _rbg_gather_table(dest, ndest) -> (ptr, src)

Group scratch positions by the output slot they contribute to: `src[ptr[j]]`
through `src[ptr[j+1]-1]` are the positions `srcpos[k]` whose `dest[k]` is `j`,
in ascending `k`. The two-argument form takes each position to contribute to
`dest` once, which is `srcpos = eachindex(dest)`.

This is what turns a scatter into a gather and so removes every atomic from
these models. Stage 2 of each accumulating callback is one thread per `j`
summing over that list, and because the list order is fixed at build time the
sum is the same on every backend. An output slot nothing contributes to gets
the zero its empty list sums to, so the stage-2 kernels assign and need no
preceding fill.

A position may appear twice with the same `dest`, and then it is added twice.
That is how an element pair whose two elemental variables are the SAME problem
variable gets its factor of two; see [`_rbg_hess_dest`](@ref).
"""
function _rbg_gather_table(srcpos::Vector{Int}, dest::Vector{Int}, ndest::Integer)
    length(srcpos) == length(dest) || throw(DimensionMismatch(
        "gather table: $(length(srcpos)) positions against $(length(dest)) " *
        "destinations"))
    cnt = zeros(Int, ndest + 1)
    for d in dest
        1 <= d <= ndest ||
            error("gather destination $d outside 1:$ndest")
        cnt[d+1] += 1
    end
    ptr = Vector{Int32}(undef, ndest + 1)
    ptr[1] = 1
    for j in 1:ndest
        ptr[j+1] = ptr[j] + cnt[j+1]
    end
    nxt = copy(ptr)
    src = Vector{Int32}(undef, length(dest))
    for (k, d) in pairs(dest)
        src[nxt[d]] = srcpos[k]
        nxt[d] += 1
    end
    return ptr, src
end

_rbg_gather_table(dest::Vector{Int}, ndest::Integer) =
    _rbg_gather_table(collect(eachindex(dest)), dest, ndest)

"""
    _rbg_var_dest(blocks, ::Val{K}) -> Vector{Int}

The variable each slot of an `K x nitem` gradient scratch array contributes
to: slot `(i-1)*K + a` of item `i` holds the derivative in variable
`blocks[i][a]`.
"""
function _rbg_var_dest(blocks, ::Val{K}) where {K}
    dest = Vector{Int}(undef, K * length(blocks))
    for (i, idx) in pairs(blocks), a in 1:K
        dest[(i-1)*K+a] = idx[a]
    end
    return dest
end

"""
    _rbg_hess_dest(blocks, ::Val{K}, hrows, hcols) -> (srcpos, dest)

Which declared Hessian entry each slot of a `K(K+1)/2 x nitem` block scratch
array contributes to, and how many times.

Slot `(a-1)*a/2 + b` of item `i`'s block, `b <= a`, holds the `(a, b)` entry of
that item's elemental Hessian, and contributes it to the problem Hessian entry
`(blocks[i][a], blocks[i][b])`, which `(hrows, hcols)` --
`_rb_hess_pattern`'s deduplicated pattern -- lists exactly once.

THE VARIABLE MAP NEED NOT BE INJECTIVE WITHIN AN ITEM: RAYBENDS' first segment
has `X1 = X2 = X3 = X(0)`. The problem Hessian entry `(i, j)` is
`sum H[a][b]` over ALL ordered pairs with `blocks[i][a] = i, blocks[i][b] = j`,
so an elemental pair `a != b` whose two variables are the SAME problem variable
lands on that variable's DIAGONAL entry from both orderings and is counted
TWICE, while the lower triangle visits it once. Such a slot is therefore listed
twice here. Omitting that factor leaves `hprod` exact -- it never reads a
coordinate -- and the assembled Hessian 96% off, so the only check that sees it
is the coordinate Hessian against `hprod` in
`test/raybends_gpu_verify.jl`.
"""
function _rbg_hess_dest(blocks, ::Val{K}, hrows, hcols) where {K}
    slot = Dict{Tuple{Int,Int},Int}(
        (r, c) => p for (p, (r, c)) in enumerate(zip(hrows, hcols)))
    nhb = K * (K + 1) ÷ 2
    srcpos = Int[]
    dest = Int[]
    sizehint!(srcpos, nhb * length(blocks))
    sizehint!(dest, nhb * length(blocks))
    for (i, idx) in pairs(blocks)
        s = 0
        for a in 1:K, b in 1:a
            s += 1
            r, c = max(idx[a], idx[b]), min(idx[a], idx[b])
            p = get(slot, (r, c), 0)
            p == 0 && error(
                "RAYBENDS_gpu: item $i's ($a, $b) entry is ($r, $c), which " *
                "the declared Hessian pattern does not contain")
            push!(srcpos, (i - 1) * nhb + s)
            push!(dest, p)
            if a != b && idx[a] == idx[b]
                push!(srcpos, (i - 1) * nhb + s)
                push!(dest, p)
            end
        end
    end
    return srcpos, dest
end

"""
    _rbg_check_indices(blocks, nvar, ::Val{K}, what)

Check that every item maps each of its `K` elemental variables into `1:nvar`.

The kernels gather `x[idx[a, i]]` with no bounds check of their own, so this is
what makes that legal, and it is also what makes the gather tables above
addressable. It runs once per build, O(nitem), against callbacks that cost
milliseconds, and it names the item and the elemental variable; the same fault
inside a kernel surfaces as an aborted launch, or as a silent read of whatever
memory is there, with nothing pointing back at the decode.
"""
function _rbg_check_indices(blocks, nvar::Integer, ::Val{K},
                            what::AbstractString) where {K}
    for (i, idx) in pairs(blocks)
        for a in 1:K
            1 <= idx[a] <= nvar || error(
                "$what $i maps elemental variable $a to index $(idx[a]), " *
                "outside 1:$nvar")
        end
    end
    return nothing
end

# Stage 2 of every accumulating callback: one thread per output slot. One
# kernel serves the gradient, the Hessian-vector product and the Hessian,
# because the three differ only in which table they are given.
@kernel function _rbg_gather_kernel!(out, @Const(sc), @Const(ptr), @Const(src))
    j = @index(Global)
    acc = zero(eltype(out))
    for p in ptr[j]:(ptr[j+1]-1)
        acc += sc[src[p]]
    end
    out[j] = acc
end

# ===========================================================================
# RAYBENDS
# ===========================================================================

"""
    RaybendsKernelData

RAYBENDS' spline segments as a struct of arrays on whatever device the model is
built for: `idx` is 8 x nel global variable indices in `RAYBENDS_EVARS` order
and `w` the element weight divided by the group scale.

`gptr`/`gsrc` gather an `8 x nel` scratch array into the gradient and
`hptr`/`hsrc` a `36 x nel` one into the declared Hessian entries; see
[`_rbg_gather_table`](@ref). A segment's 8 entries are contiguous in `idx`'s
first dimension, which is the order a thread reads them in.
"""
struct RaybendsKernelData{MI,VI,VT}
    idx::MI
    w::VT
    gptr::VI
    gsrc::VI
    hptr::VI
    hsrc::VI
    nel::Int
end

function _rbg_raybends_data(els, blocks, ::Type{T}, nvar, hr, hc, backend) where {T}
    nel = length(els)
    idxh = Matrix{Int32}(undef, 8, nel)
    wh = Vector{T}(undef, nel)
    for (i, e) in pairs(els)
        for a in 1:8
            idxh[a, i] = e.idx[a]
        end
        wh[i] = e.w
    end
    gptr, gsrc = _rbg_gather_table(_rbg_var_dest(blocks, Val(8)), nvar)
    hptr, hsrc = _rbg_gather_table(_rbg_hess_dest(blocks, Val(8), hr, hc)...,
                                   length(hr))
    cv = v -> ExaModels.convert_array(v, backend)
    return RaybendsKernelData(cv(idxh), cv(wh), cv(gptr), cv(gsrc),
                              cv(hptr), cv(hsrc), nel)
end

# Kernels. Every `idx` entry lands in `1:length(x)`, which the gathers rely on:
# `_rbg_check_indices` establishes it once at build time, where it can name the
# segment and the elemental variable that went wrong.

@kernel function _rbg_rb_val_kernel!(buf, @Const(x), @Const(idx), @Const(w), tab)
    i = @index(Global)
    E = eltype(buf)
    buf[i] = w[i] * raybends_time(ntuple(a -> E(x[idx[a, i]]), Val(8)), tab)
end

@kernel function _rbg_rb_gradblock_kernel!(sc, @Const(x), @Const(idx), @Const(w), tab)
    i = @index(Global)
    E = eltype(sc)
    g = raybends_grad(ntuple(a -> E(x[idx[a, i]]), Val(8)), tab)
    wi = w[i]
    base = (i - 1) * 8
    ntuple(a -> (sc[base+a] = wi * g[a]; nothing), Val(8))
end

@kernel function _rbg_rb_hvpblock_kernel!(
        sc, @Const(x), @Const(vv), @Const(idx), @Const(w), tab)
    i = @index(Global)
    E = eltype(sc)
    h = raybends_hvp(ntuple(a -> E(x[idx[a, i]]), Val(8)),
                     ntuple(a -> E(vv[idx[a, i]]), Val(8)), tab)
    wi = w[i]
    base = (i - 1) * 8
    ntuple(a -> (sc[base+a] = wi * h[a]; nothing), Val(8))
end

# One column of one segment's 8x8 block per thread. Slot `(r-1)*r/2 + j` of the
# block is its (r, j) entry, `r >= j`, which is the order `_rbg_hess_dest`
# assumes. Unrolled over the rows with the triangle as a predicate so the
# Hessian column stays in registers: a tuple indexed by a loop variable would
# have to be spilled to an addressable stack slot.
@kernel function _rbg_rb_hessblock_kernel!(sc, @Const(x), @Const(idx), @Const(w), tab)
    j, i = @index(Global, NTuple)
    E = eltype(sc)
    h = raybends_hvp(ntuple(a -> E(x[idx[a, i]]), Val(8)),
                     ntuple(a -> ifelse(a == j, one(E), zero(E)), Val(8)), tab)
    wi = w[i]
    base = (i - 1) * 36
    ntuple(Val(8)) do r
        r >= j && (sc[base+((r-1)*r)÷2+j] = wi * h[r])
        nothing
    end
end

"""
    raybends_kernel_model(; T = Float64, backend = nothing, single = true) -> ExaModel

Build RAYBENDS as an `ExaModel` whose objective callbacks are
KernelAbstractions kernels over [`raybends_time`](@ref), with the per-segment
data resident on `backend`.

`backend = nothing` builds a host model that runs the same kernels on
KernelAbstractions' CPU backend; `backend = CUDABackend()` builds a device
model that never indexes a device array from the host, so it is usable with
`CUDA.allowscalar(false)`.

Unconstrained, 2050 variables; one `ScalarNonlinearOracle` holds all 1026
spline segments, with `f`, `grad!`, `hvp!` and an explicit `hess!`. `single`
selects the reading of the Fortran's four inexact bare constants exactly as in
[`raybends_model`](@ref), and `true` -- the default -- is the one that
reproduces CUTEst.

No kernel here uses `KernelAbstractions.@atomic`, so every callback is bitwise
reproducible across backends and `T` may be a non-bits type -- on
`backend = nothing`, which is the only backend a `BigFloat` can live on.

The declared Hessian pattern is the deduplicated union, identical to
[`raybends_model`](@ref)'s and to CUTEst's `nnzh`, entry for entry and in the
same order: 15351 entries, every one of them distinct. `hess!` therefore
delivers each entry already accumulated over the segments that touch it.
"""
function raybends_kernel_model(; T = Float64, backend = nothing,
                               single::Bool = true)
    P, els = _raybends_decode()
    tab = raybends_table(T; single)
    n = length(P.varnames)
    blocks = [e.idx for e in els]
    _rbg_check_indices(blocks, n, Val(8), "RAYBENDS: segment")
    hr, hc = _rb_hess_pattern(blocks, 8)
    d = _rbg_raybends_data(els, blocks, T, n, hr, hc, backend)
    ka = backend === nothing ? KernelAbstractions.CPU() : backend
    nel = d.nel

    # The decode's indices are absolute positions in `1:n`, so a gather against
    # an offset array would read the wrong variables silently.
    valbuf = ExaModels.convert_array(zeros(T, nel), backend)
    gsc = ExaModels.convert_array(zeros(T, 8 * nel), backend)
    hsc = ExaModels.convert_array(zeros(T, 36 * nel), backend)

    f = xv -> begin
        Base.require_one_based_indexing(xv)
        _rbg_rb_val_kernel!(ka)(valbuf, xv, d.idx, d.w, tab; ndrange = nel)
        KernelAbstractions.synchronize(ka)
        return sum(valbuf)
    end
    grad! = (gv, xv) -> begin
        Base.require_one_based_indexing(gv, xv)
        _rbg_rb_gradblock_kernel!(ka)(gsc, xv, d.idx, d.w, tab; ndrange = nel)
        KernelAbstractions.synchronize(ka)
        _rbg_gather_kernel!(ka)(gv, gsc, d.gptr, d.gsrc; ndrange = n)
        KernelAbstractions.synchronize(ka)
        return gv
    end
    hvp! = (Hv, xv, vv) -> begin
        Base.require_one_based_indexing(Hv, xv, vv)
        _rbg_rb_hvpblock_kernel!(ka)(gsc, xv, vv, d.idx, d.w, tab; ndrange = nel)
        KernelAbstractions.synchronize(ka)
        _rbg_gather_kernel!(ka)(Hv, gsc, d.gptr, d.gsrc; ndrange = n)
        KernelAbstractions.synchronize(ka)
        return Hv
    end
    hess! = (vals, xv) -> begin
        Base.require_one_based_indexing(vals, xv)
        _rbg_rb_hessblock_kernel!(ka)(hsc, xv, d.idx, d.w, tab;
                                      ndrange = (8, nel))
        KernelAbstractions.synchronize(ka)
        _rbg_gather_kernel!(ka)(vals, hsc, d.hptr, d.hsrc; ndrange = length(hr))
        KernelAbstractions.synchronize(ka)
        return vals
    end

    c = ExaModels.ExaCore(T; backend = backend)
    c, _ = ExaModels.add_var(c, n; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    c = ExaModels.objective(c, ExaModels.ScalarNonlinearOracle(;
        nvar = n, f = f, grad! = grad!, hvp! = hvp!, hess! = hess!,
        hess_rows = hr, hess_cols = hc, adapt = Val(false)))
    return ExaModels.ExaModel(c)
end

# ===========================================================================
# BLEACHNG
# ===========================================================================

"""
    BleachngKernelData

BLEACHNG's observation groups as a struct of arrays on whatever device the
model is built for: `idx` is 5 x nobs global variable indices in
`BLEACHNG_EVARS` order, `par` the 5 x nobs element parameters
`(VOLGAS, VOLLIQ, H, T, REALID-1)`, `w` the element weight and `z` the
measurement the group's constant carries.

`gptr`/`gsrc` gather a `5 x nobs` scratch array into the gradient and
`hptr`/`hsrc` a `15 x nobs` one into the declared Hessian entries; see
[`_rbg_gather_table`](@ref).
"""
struct BleachngKernelData{MI,VI,MT,VT}
    idx::MI
    par::MT
    w::VT
    z::VT
    gptr::VI
    gsrc::VI
    hptr::VI
    hsrc::VI
    nobs::Int
end

function _rbg_bleachng_data(obs, blocks, ::Type{T}, nvar, hr, hc, backend) where {T}
    nobs = length(obs)
    idxh = Matrix{Int32}(undef, 5, nobs)
    parh = Matrix{T}(undef, 5, nobs)
    wh = Vector{T}(undef, nobs)
    zh = Vector{T}(undef, nobs)
    for (i, o) in pairs(obs)
        for a in 1:5
            idxh[a, i] = o.idx[a]
            parh[a, i] = o.par[a]
        end
        wh[i] = o.w
        zh[i] = o.z
    end
    gptr, gsrc = _rbg_gather_table(_rbg_var_dest(blocks, Val(5)), nvar)
    hptr, hsrc = _rbg_gather_table(_rbg_hess_dest(blocks, Val(5), hr, hc)...,
                                   length(hr))
    cv = v -> ExaModels.convert_array(v, backend)
    return BleachngKernelData(cv(idxh), cv(parh), cv(wh), cv(zh), cv(gptr),
                              cv(gsrc), cv(hptr), cv(hsrc), nobs)
end

# Kernels. One observation contributes `(w * conc - z)^2`, so a derivative of it
# needs the value as well as the derivative of `conc`:
#
#     grad  = 2 (w conc - z) w grad(conc)
#     hess  = 2 w^2 grad(conc) grad(conc)' + 2 (w conc - z) w hess(conc)
#
# Every `idx` entry lands in `1:length(x)`, which the gathers rely on:
# `_rbg_check_indices` establishes it once at build time.

@kernel function _rbg_bl_val_kernel!(
        buf, @Const(x), @Const(idx), @Const(par), @Const(w), @Const(z))
    i = @index(Global)
    E = eltype(buf)
    p = ntuple(a -> par[a, i], Val(5))
    r = w[i] * bleachng_conc(ntuple(a -> E(x[idx[a, i]]), Val(5)), p) - z[i]
    buf[i] = r * r
end

@kernel function _rbg_bl_gradblock_kernel!(
        sc, @Const(x), @Const(idx), @Const(par), @Const(w), @Const(z))
    i = @index(Global)
    E = eltype(sc)
    p = ntuple(a -> par[a, i], Val(5))
    val, g = bleachng_grad(ntuple(a -> E(x[idx[a, i]]), Val(5)), p)
    wi = w[i]
    r2 = 2 * (wi * val - z[i]) * wi
    base = (i - 1) * 5
    ntuple(a -> (sc[base+a] = r2 * g[a]; nothing), Val(5))
end

@kernel function _rbg_bl_hvpblock_kernel!(
        sc, @Const(x), @Const(vv), @Const(idx), @Const(par), @Const(w), @Const(z))
    i = @index(Global)
    E = eltype(sc)
    p = ntuple(a -> par[a, i], Val(5))
    dd = ntuple(a -> E(vv[idx[a, i]]), Val(5))
    val, g, hd = bleachng_grad_hvp(ntuple(a -> E(x[idx[a, i]]), Val(5)), dd, p)
    wi = w[i]
    r2 = 2 * (wi * val - z[i]) * wi
    c1 = 2 * wi * wi * _rbg_tdot(g, dd)
    base = (i - 1) * 5
    ntuple(a -> (sc[base+a] = c1 * g[a] + r2 * hd[a]; nothing), Val(5))
end

# One column of one observation's 5x5 block per thread. `g[j]` is read as the
# dot of the gradient with the column's own `e_j` rather than by indexing with
# `j`: a tuple indexed by a run-time value would have to be spilled to an
# addressable stack slot.
@kernel function _rbg_bl_hessblock_kernel!(
        sc, @Const(x), @Const(idx), @Const(par), @Const(w), @Const(z))
    j, i = @index(Global, NTuple)
    E = eltype(sc)
    p = ntuple(a -> par[a, i], Val(5))
    ej = ntuple(a -> ifelse(a == j, one(E), zero(E)), Val(5))
    val, g, hd = bleachng_grad_hvp(ntuple(a -> E(x[idx[a, i]]), Val(5)), ej, p)
    wi = w[i]
    r2 = 2 * (wi * val - z[i]) * wi
    c1 = 2 * wi * wi * _rbg_tdot(g, ej)
    base = (i - 1) * 15
    ntuple(Val(5)) do r
        r >= j && (sc[base+((r-1)*r)÷2+j] = c1 * g[r] + r2 * hd[r])
        nothing
    end
end

"""
    bleachng_kernel_model(; T = Float64, backend = nothing) -> ExaModel

Build BLEACHNG as an `ExaModel` whose objective callbacks are
KernelAbstractions kernels over [`bleachng_conc`](@ref) -- the closed-form
solution of the ODE the appended ODESSA integrates numerically -- with the
per-observation data resident on `backend`.

`backend = nothing` builds a host model that runs the same kernels on
KernelAbstractions' CPU backend; `backend = CUDABackend()` builds a device
model that never indexes a device array from the host, so it is usable with
`CUDA.allowscalar(false)`.

Unconstrained, 17 variables. The 1602 observation groups go into one
`ScalarNonlinearOracle` with `f`, `grad!`, `hvp!` and an explicit `hess!`; the
1602 purely linear groups stay an ordinary SIMD ExaModels objective, since they
contain no external function.

`prod = true` is the default because mixing that SIMD objective with an oracle
is what makes it necessary: ExaModels' KernelAbstractions override of `hprod!`
for a model with oracles dispatches on the prodhelper `prod = true` builds, and
without it the call falls back to the generic method, which evaluates the SIMD
objective's Hessian through host loops over the device arrays --
`Scalar indexing is disallowed`. [`raybends_kernel_model`](@ref) has no SIMD
part, its objective being an oracle in full, and so does not need it.

No kernel here uses `KernelAbstractions.@atomic`, so every callback is bitwise
reproducible across backends and `T` may be a non-bits type -- on
`backend = nothing`, which is the only backend a `BigFloat` can live on.

The declared Hessian pattern is the deduplicated union, identical to
[`bleachng_model`](@ref)'s, entry for entry and in the same order; `hess!`
delivers each entry already accumulated over the observations that touch it.
The model's `nnzh` is 1659 -- 57 from the oracle plus one diagonal entry per
linear group -- and denotes 57 distinct coordinates, which is CUTEst's `nnzh`.
CUTEst cannot adjudicate either model here: its gradient for this problem is
wrong in the `QIN` columns and its Hessian is the Gauss-Newton part built from
that gradient (see the header of RAYBENDS.jl).
"""
function bleachng_kernel_model(; T = Float64, backend = nothing,
                               prod::Bool = true)
    P, obs, lins = _bleachng_decode(T)
    n = length(P.varnames)
    blocks = [o.idx for o in obs]
    _rbg_check_indices(blocks, n, Val(5), "BLEACHNG: observation")
    hr, hc = _rb_hess_pattern(blocks, 5)
    d = _rbg_bleachng_data(obs, blocks, T, n, hr, hc, backend)
    ka = backend === nothing ? KernelAbstractions.CPU() : backend
    nobs = d.nobs

    valbuf = ExaModels.convert_array(zeros(T, nobs), backend)
    gsc = ExaModels.convert_array(zeros(T, 5 * nobs), backend)
    hsc = ExaModels.convert_array(zeros(T, 15 * nobs), backend)

    f = xv -> begin
        Base.require_one_based_indexing(xv)
        _rbg_bl_val_kernel!(ka)(valbuf, xv, d.idx, d.par, d.w, d.z;
                                ndrange = nobs)
        KernelAbstractions.synchronize(ka)
        return sum(valbuf)
    end
    grad! = (gv, xv) -> begin
        Base.require_one_based_indexing(gv, xv)
        _rbg_bl_gradblock_kernel!(ka)(gsc, xv, d.idx, d.par, d.w, d.z;
                                      ndrange = nobs)
        KernelAbstractions.synchronize(ka)
        _rbg_gather_kernel!(ka)(gv, gsc, d.gptr, d.gsrc; ndrange = n)
        KernelAbstractions.synchronize(ka)
        return gv
    end
    hvp! = (Hv, xv, vv) -> begin
        Base.require_one_based_indexing(Hv, xv, vv)
        _rbg_bl_hvpblock_kernel!(ka)(gsc, xv, vv, d.idx, d.par, d.w, d.z;
                                     ndrange = nobs)
        KernelAbstractions.synchronize(ka)
        _rbg_gather_kernel!(ka)(Hv, gsc, d.gptr, d.gsrc; ndrange = n)
        KernelAbstractions.synchronize(ka)
        return Hv
    end
    hess! = (vals, xv) -> begin
        Base.require_one_based_indexing(vals, xv)
        _rbg_bl_hessblock_kernel!(ka)(hsc, xv, d.idx, d.par, d.w, d.z;
                                      ndrange = (5, nobs))
        KernelAbstractions.synchronize(ka)
        _rbg_gather_kernel!(ka)(vals, hsc, d.hptr, d.hsrc; ndrange = length(hr))
        KernelAbstractions.synchronize(ka)
        return vals
    end

    c = ExaModels.ExaCore(T; backend = backend)
    c, x = ExaModels.add_var(c, n; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    if !isempty(lins)
        data = [(a = l[1], j = l[2], z = l[3]) for l in lins]
        c, _ = ExaModels.add_obj(c,
            Base.Generator(q -> (q.a * x[q.j] - q.z)^2, data))
    end
    c = ExaModels.objective(c, ExaModels.ScalarNonlinearOracle(;
        nvar = n, f = f, grad! = grad!, hvp! = hvp!, hess! = hess!,
        hess_rows = hr, hess_cols = hc, adapt = Val(false)))
    return ExaModels.ExaModel(c; prod = prod)
end
