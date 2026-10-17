# CAR2 / ORBIT2 as ExaModels whose `Ktyp` rows are filled by
# KernelAbstractions kernels over `colloc`, with the per-element interval data
# resident on the model's backend.
#
# `colloc_model` (COLLOC.jl) interfaces the same `colloc` through host loops
# over a shared scratch vector. This file's kernels let the model be built and
# evaluated with scalar indexing of device arrays disabled
# (`CUDA.allowscalar(false)`), which host loops over a device array cannot be.
# Both builders share `_colloc_decode` and `_colloc_simd_core`, so the SIF
# reading, the bucketing and the non-external part have one definition.
#
# WHY A KERNEL AND NOT AN ExaModels EXPRESSION. Unchanged from COLLOC.jl: one
# ORBIT2 row is three FSYS evaluations, two of them through a `^1.5`, about 75
# elementary operations, and ExaModels nests a type parameter per operation.
# The generic function is the port; this file is only the launch wiring.
#
# WHAT IS PARALLEL. One collocation element per thread for the row value and
# the Jacobian: 3996 elements over 5999 variables for CAR2, 1794 over 2698 for
# ORBIT2, each element reading exactly its `2*NX + NU + 1` elemental
# variables. The Hessian runs one thread per (element, column) instead, as in
# `JIMACK_gpu.jl`: a column of an element block is one directional derivative
# of the element gradient, the columns are independent, and one thread per
# element alone is 125 warps for CAR2 and 56 for ORBIT2, too few to fill a
# device with 80 multiprocessors.
#
# THE EQUATION INDEX IS A COMPILE-TIME CONSTANT, which is why there is one
# launch per distinct `ID` rather than one over all elements. `colloc` reads
# `x[id]`, `f[id]` and `fc[id]` out of `NTuple`s; with `id` a value loaded from
# an array those tuples would have to be spilled to an addressable stack slot
# -- for the nested `Dual`s of the Hessian sweep, hundreds of bytes of local
# memory per thread. `CollocShape{NX,NU,ID}` carries the index in a type
# parameter, so each launch specializes on it and every tuple access is a
# `getfield`. The element positions for each `ID` are precomputed, so the
# launches partition the elements and every kernel still addresses its output
# by the element's position in the DECODE order -- the Jacobian and Hessian
# slot layouts are therefore identical to `colloc_model`'s, not a permutation
# of them.
#
# DERIVATIVES. ForwardDiff inside the kernel, over `colloc`, with the sweeps
# the host port's `ForwardDiff.gradient!`/`hessian!` would use but seeded by
# hand: the configs those take allocate a result and consult a mutable tag
# counter, neither of which is available in a kernel. First order is one level
# with `2*NX + NU + 1` partials; a Hessian column is one 1-partial level over
# that, 2 x 11 (CAR2) or 2 x 16 (ORBIT2) scalars per intermediate value, which
# is what the whole `nv x nv` Hessian in one sweep would not be.
#
# NO ATOMICS ANYWHERE, so every callback is bitwise reproducible across
# backends and the builder serves a non-bits element type.
#
#   * The row values are a disjoint write: each `Ktyp` element owns one
#     constraint row, which `_colloc_check_rows` establishes at build time.
#   * The Jacobian and the Hessian are disjoint writes: each element owns a
#     contiguous slice of the value array, and the duplicate `(row, col)`
#     coordinates across elements are summed by whoever assembles the matrix.
#   * The Hessian-VECTOR product is the one scatter -- every element touches
#     the free time Q, so all 3996 of CAR2's contribute to that one component
#     -- and it runs as two kernels instead of one atomic one: the first
#     writes each element's `nv` contributions into its own slice of a
#     scratch array, the second sums, one thread per VARIABLE, over the
#     contributions listed for it in a table built once at model build time.
#     The summation order is that table's and so is the same on every backend.
#
# WHY `hvp!` IS SUPPLIED AND NOT LEFT TO THE COORDINATE PATH. Without it
# ExaModels' `hprod!` for an `OracleEvaluator` falls back to accumulating
# `hess!`'s coordinates on the HOST, element by element over `nnzh` -- 265736
# scalar reads of a device array for CAR2, which `CUDA.allowscalar(false)`
# refuses outright. Supplying `hvp!` also makes `hprod` independent of the
# declared sparsity, which is what lets the two be compared against each other
# (`test/colloc_gpu_verify.jl`).
#
# HESSIAN PATTERN: BLOCKWISE WITH DUPLICATES, the lower triangle of each
# element's own `nv x nv` block in the slot order the kernel writes --
# `nv*(nv+1)/2 * nel`, which is 66 x 3996 for CAR2 and 136 x 1794 for ORBIT2.
# It is the pattern `colloc_model` declares too, entry for entry and in the
# same order, and the duplicates are intended: Q appears in every element's
# block, and the repetition is how its entry accumulates every element's
# contribution, since `hess!` writes one value per declared entry and the
# consumer sums duplicates.
#
# Two things follow, and they are the reasons for the choice. `hess!` is a
# single pass of pure disjoint writes, with no gather stage and so nothing for
# a scheduler to reorder. And because the two builders declare the same
# entries in the same order, their coordinate values are comparable slot by
# slot rather than only as assembled matrices -- the stronger reading, since it
# sees a fill that lands an element's values in the wrong slots of its own
# block. What the pattern costs is that `nnzh` far exceeds the number of
# distinct nonzeros, and that it is only correct for an element whose variable
# map is injective, which `_colloc_check_indices` requires rather than
# assumes.
#
# WHY THE MODEL IS BUILT WITH `prod = true`. Mixing a SIMD symbolic part with
# an oracle is what makes it necessary, and both problems do: the
# control-magnitude rows and ORBIT2's terminal rows are ordinary SIMD
# expressions. ExaModels' KernelAbstractions override of `hprod!` for a model
# with oracles dispatches on the prodhelper that `prod = true` builds, and
# without it the call falls back to the generic method, which evaluates the
# SIMD part's Hessian through host loops over the device arrays --
# `Scalar indexing is disallowed`. `src/native/HS88_92.jl` carries the same
# note for the same reason.
#
# ONE BUILDER FOR BOTH BACKENDS. `backend = nothing` runs the same kernels on
# KernelAbstractions' CPU backend. A CPU-versus-GPU comparison therefore tests
# the device rather than two unrelated implementations;
# `test/colloc_gpu_verify.jl` is that comparison.

using KernelAbstractions

# ---------------------------------------------------------------------------
# The element shape in the type domain.

"""
    CollocShape{NX,NU,ID}

One `Ktyp` element's shape as type parameters: `NX` states, `NU` controls and
the equation index `ID` whose defect the element evaluates.

`ID` is a type parameter rather than a field because `colloc` indexes
`NTuple`s with it; see the header.
"""
struct CollocShape{NX,NU,ID} end

"Number of elemental variables of one element."
@inline collocnv(::CollocShape{NX,NU}) where {NX,NU} = 2 * NX + NU + 1

"The same count as a `Val`, for the `ntuple`s a kernel gathers into."
@inline _colloc_nvval(::CollocShape{NX,NU}) where {NX,NU} = Val(2 * NX + NU + 1)

"Number of lower-triangle slots of one element's dense Hessian block."
@inline _colloc_nhb(sh::CollocShape) = (nv = collocnv(sh); nv * (nv + 1) ÷ 2)

"""
    _colloc_zval(sys, z, t, t1, shape) -> value

[`colloc`](@ref) on the element's own `2*NX + NU + 1` elemental variables, in
the slot order [`collocvar`](@ref) uses: `NX` initial states, `NX` final
states, `NU` controls, then the free time.

The same split `CollocRowFun` performs, over a tuple rather than a scratch
vector and with the equation index taken from the shape's type parameter.
"""
@inline function _colloc_zval(sys, z::NTuple{NV,Any}, t, t1,
                              ::CollocShape{NX,NU,ID}) where {NV,NX,NU,ID}
    x = ntuple(k -> z[k], Val(NX))
    xp = ntuple(k -> z[NX+k], Val(NX))
    u = ntuple(k -> z[2NX+k], Val(NU))
    return colloc(sys, x, xp, u, z[2NX+NU+1], t, t1, ID)
end

# ForwardDiff seeding, written out rather than going through
# ForwardDiff.gradient/hessian: those take an `AbstractVector`, allocate a
# result, and consult a mutable tag counter, none of which is available in a
# kernel. Seeding by hand keeps everything in registers.

"""
    _colloc_valgrad(sys, z, t, t1, shape) -> (value, gradient::NTuple{NV})

One element's row value and its gradient in the element's own variables, by
one level of forward-mode AD over [`_colloc_zval`](@ref).
"""
@inline function _colloc_valgrad(sys, z::NTuple{NV,T}, t, t1,
                                 sh::CollocShape) where {NV,T}
    D = ForwardDiff.Dual{:COLLOC,T,NV}
    zd = ntuple(i -> D(z[i], ForwardDiff.Partials(
                    ntuple(j -> ifelse(i == j, one(T), zero(T)), Val(NV)))),
                Val(NV))
    r = _colloc_zval(sys, zd, t, t1, sh)
    return ForwardDiff.value(r), ntuple(i -> ForwardDiff.partials(r, i), Val(NV))
end

"""
    _colloc_hvp(sys, z, d, t, t1, shape) -> NTuple{NV}

`hess(row)(z) * d` for one element, as the directional derivative of the
gradient: one `NV`-partial sweep over one 1-partial sweep. With `d = e_j` this
is column `j` of the element's Hessian block, which is how the Hessian
coordinate kernel reads it.
"""
@inline function _colloc_hvp(sys, z::NTuple{NV,T}, d::NTuple{NV,T}, t, t1,
                             sh::CollocShape) where {NV,T}
    O = ForwardDiff.Dual{:COLLOCdir,T,1}
    zo = ntuple(i -> O(z[i], ForwardDiff.Partials((d[i],))), Val(NV))
    _, g = _colloc_valgrad(sys, zo, t, t1, sh)
    return ntuple(i -> ForwardDiff.partials(g[i], 1), Val(NV))
end

# ---------------------------------------------------------------------------
# Device-resident element data.

"""
    CollocKernelData

All of a problem's `Ktyp` elements as a struct of arrays on whatever device
the model is built for: `idx` is `nv x nel` global variable indices in
[`collocvar`](@ref) slot order, `row` the constraint row each element fills,
`t`/`t1` the interval endpoints and `w` the element weight divided by the
group scale.

`pos` holds, for each equation index `1:NX`, the positions in `1:nel` of the
elements with that index -- one launch per entry, see the header. `vptr`/`vsrc`
are the scatter table the Hessian-vector product sums through: `vsrc[p]` for
`p in vptr[j]:vptr[j+1]-1` are the flat positions in an `nv x nel` scratch
array that contribute to variable `j`.

An element's `nv` entries are contiguous in `idx`'s first dimension, which is
the order a thread reads them in.
"""
struct CollocKernelData{MI,VI,VT,TP}
    idx::MI
    row::VI
    t::VT
    t1::VT
    w::VT
    pos::TP
    vptr::VI
    vsrc::VI
    nel::Int
end

"""
    _colloc_check_indices(elems, nvar, nv)

Check that every element maps its `nv` elemental variable slots to `nv`
DISTINCT indices in `1:nvar`.

Two kernel properties rest on this. The gathers `x[idx[k, i]]` have no bounds
check of their own, and on a device an out-of-range gather has no useful
diagnosis -- it reads whatever memory is there. And the blockwise Hessian
pattern is only right for an injective map: the problem Hessian entry `(i, j)`
sums the element Hessian over every ordered slot pair landing on it, so a slot
pair `a != b` whose two slots were the same variable would have to be counted
twice on that variable's diagonal, where the lower triangle visits it once.
`collocvar` maps the slots to the states at the two interval ends, the
interval's controls and the free time, which the SIF declares as distinct
variables; a SIF that identified two of them would need the Hessian assembled
differently and is refused rather than assembled wrongly.

It runs once per build, O(nel), against callbacks that cost milliseconds, and
it names the element and the slot; the same fault inside a kernel surfaces as
an aborted launch, or as a silently wrong answer, with nothing pointing back at
the decode that produced the index.
"""
function _colloc_check_indices(elems, nvar::Integer, nv::Integer)
    for (i, e) in pairs(elems)
        for k in 1:nv
            j = collocvar(e, k)
            1 <= j <= nvar || error(
                "COLLOC: element $i maps elemental variable slot $k to index " *
                "$j, outside 1:$nvar")
            for l in 1:(k-1)
                collocvar(e, l) == j && error(
                    "COLLOC: element $i maps elemental variable slots $l and " *
                    "$k to the same variable index $j; the blockwise Hessian " *
                    "pattern assumes they are distinct")
            end
        end
    end
    return nothing
end

"""
    _colloc_check_rows(elems, ncon)

Check that the elements fill distinct constraint rows in `1:ncon`.

The value kernel ASSIGNS `res[row[i]]`, one thread per element, which is a
disjoint write only if no two elements share a row. The SIF makes one group per
(interval, equation) pair, so they do not; a SIF that put two external elements
in one group would need an accumulation here, and it is better to be refused
than to have one element's value silently overwrite the other's.
"""
function _colloc_check_rows(elems, ncon::Integer)
    seen = falses(ncon)
    for (i, e) in pairs(elems)
        1 <= e.row <= ncon ||
            error("COLLOC: element $i fills row $(e.row), outside 1:$ncon")
        seen[e.row] &&
            error("COLLOC: row $(e.row) is filled by more than one Ktyp " *
                  "element; the kernels assume one element per row")
        seen[e.row] = true
    end
    return nothing
end

"""
    _colloc_scatter_table(elems, nvar, nv) -> (vptr, vsrc)

The variable-wise transpose of the element-variable map: `vsrc[vptr[j]]`
through `vsrc[vptr[j+1]-1]` are the flat positions `(i-1)*nv + k` of every
element slot that holds variable `j`.

This is what lets the Hessian-vector product accumulate without an atomic: the
scatter becomes a gather, one thread per variable, summing in the order this
table lists -- the same order on every backend.
"""
function _colloc_scatter_table(elems, nvar::Integer, nv::Integer)
    cnt = zeros(Int, nvar + 1)
    for e in elems, k in 1:nv
        cnt[collocvar(e, k)+1] += 1
    end
    vptr = Vector{Int32}(undef, nvar + 1)
    vptr[1] = 1
    for j in 1:nvar
        vptr[j+1] = vptr[j] + cnt[j+1]
    end
    fill = copy(vptr)
    vsrc = Vector{Int32}(undef, nv * length(elems))
    for (i, e) in pairs(elems), k in 1:nv
        j = collocvar(e, k)
        vsrc[fill[j]] = (i - 1) * nv + k
        fill[j] += 1
    end
    return vptr, vsrc
end

function _colloc_kernel_data(elems, ::Type{T}, nvar, nv, ::Val{NX},
                             backend) where {T,NX}
    nel = length(elems)
    idxh = Matrix{Int32}(undef, nv, nel)
    rowh = Vector{Int32}(undef, nel)
    th = Vector{T}(undef, nel)
    t1h = Vector{T}(undef, nel)
    wh = Vector{T}(undef, nel)
    for (i, e) in pairs(elems)
        for k in 1:nv
            idxh[k, i] = collocvar(e, k)
        end
        rowh[i] = e.row
        th[i] = e.t
        t1h[i] = e.t1
        wh[i] = e.w
    end
    posh = ntuple(id -> Int32[i for (i, e) in pairs(elems) if e.id == id], Val(NX))
    vptr, vsrc = _colloc_scatter_table(elems, nvar, nv)
    cv = v -> ExaModels.convert_array(v, backend)
    return CollocKernelData(cv(idxh), cv(rowh), cv(th), cv(t1h), cv(wh),
                            map(cv, posh), cv(vptr), cv(vsrc), nel)
end

# ---------------------------------------------------------------------------
# Kernels.
#
# Every `idx` entry lands in `1:length(x)` and every `row` entry in
# `1:length(res)`, which the gathers and the row write below rely on:
# `_colloc_check_indices` and `_colloc_check_rows` establish both once at
# build time, where they can name the element and the slot that went wrong.
#
# `pos[p]` is the position in the decode order of the `p`-th element with this
# launch's equation index, so an output slot is addressed by `i = pos[p]` and
# the declared sparsity stays in decode order.

@kernel function _colloc_cons_kernel!(
        res, @Const(x), @Const(pos), @Const(idx), @Const(row), @Const(t),
        @Const(t1), @Const(w), sys, sh)
    p = @index(Global)
    i = pos[p]
    E = eltype(res)
    z = ntuple(k -> E(x[idx[k, i]]), _colloc_nvval(sh))
    res[row[i]] = w[i] * _colloc_zval(sys, z, t[i], t1[i], sh)
end

@kernel function _colloc_jac_kernel!(
        vals, @Const(x), @Const(pos), @Const(idx), @Const(t), @Const(t1),
        @Const(w), sys, sh)
    p = @index(Global)
    i = pos[p]
    E = eltype(vals)
    nvv = _colloc_nvval(sh)
    z = ntuple(k -> E(x[idx[k, i]]), nvv)
    _, g = _colloc_valgrad(sys, z, t[i], t1[i], sh)
    wi = w[i]
    base = (i - 1) * collocnv(sh)
    ntuple(a -> (vals[base+a] = wi * g[a]; nothing), nvv)
end

# One column of one element's Hessian block per thread, as the element's
# Hessian-vector product in the direction of that column. Slot
# `(r - 1) * r / 2 + j` of the block is its (r, j) entry, `r >= j`, which is
# the order `colloc_model` declares and `_colloc_hess_structure!` repeats.
@kernel function _colloc_hess_kernel!(
        vals, @Const(x), @Const(pos), @Const(idx), @Const(row), @Const(t),
        @Const(t1), @Const(w), @Const(mult), sys, sh)
    j, p = @index(Global, NTuple)
    i = pos[p]
    E = eltype(vals)
    nvv = _colloc_nvval(sh)
    z = ntuple(k -> E(x[idx[k, i]]), nvv)
    h = _colloc_hvp(sys, z,
                    ntuple(k -> ifelse(k == j, one(E), zero(E)), nvv),
                    t[i], t1[i], sh)
    s = w[i] * mult[row[i]]
    base = (i - 1) * _colloc_nhb(sh)
    # Unrolled over the block's rows, with the triangle as a predicate, so the
    # Hessian column stays in registers: a tuple indexed by a loop variable
    # would have to be spilled to an addressable stack slot.
    ntuple(nvv) do r
        r >= j && (vals[base+((r-1)*r)÷2+j] = s * h[r])
        nothing
    end
end

# Hessian-vector product, stage one: each element's `nv` contributions to
# `(sum_i y_i nabla^2 c_i) v`, into its own slice of the scratch array.
@kernel function _colloc_hvpblock_kernel!(
        sc, @Const(x), @Const(vv), @Const(pos), @Const(idx), @Const(row),
        @Const(t), @Const(t1), @Const(w), @Const(mult), sys, sh)
    p = @index(Global)
    i = pos[p]
    E = eltype(sc)
    nvv = _colloc_nvval(sh)
    z = ntuple(k -> E(x[idx[k, i]]), nvv)
    d = ntuple(k -> E(vv[idx[k, i]]), nvv)
    h = _colloc_hvp(sys, z, d, t[i], t1[i], sh)
    s = w[i] * mult[row[i]]
    base = (i - 1) * collocnv(sh)
    ntuple(a -> (sc[base+a] = s * h[a]; nothing), nvv)
end

# Stage two: one thread per VARIABLE, summing the scratch entries listed for it
# by `_colloc_scatter_table`. Assigns, so it needs no preceding fill, and a
# variable no element touches gets the zero its empty list sums to.
@kernel function _colloc_gather_kernel!(out, @Const(sc), @Const(vptr), @Const(vsrc))
    j = @index(Global)
    acc = zero(eltype(out))
    for p in vptr[j]:(vptr[j+1]-1)
        acc += sc[vsrc[p]]
    end
    out[j] = acc
end

# ---------------------------------------------------------------------------
# Assembly

"""
    _colloc_jac_structure!(rows, cols, elems, nv)

The Jacobian pattern: each element's row against each of its `nv` elemental
variables, in decode order, which is the slot order
[`_colloc_jac_kernel!`](@ref) writes.
"""
function _colloc_jac_structure!(rows, cols, elems, nv::Integer)
    for e in elems, k in 1:nv
        push!(rows, e.row)
        push!(cols, collocvar(e, k))
    end
    return nothing
end

"""
    _colloc_hess_structure!(rows, cols, elems, nv)

The Hessian pattern: the lower triangle of each element's dense `nv x nv`
block, in decode order. Pairs repeat across elements (Q sits in every one) and
that is intended; see the header.
"""
function _colloc_hess_structure!(rows, cols, elems, nv::Integer)
    for e in elems, a in 1:nv, b in 1:a
        i, j = collocvar(e, a), collocvar(e, b)
        push!(rows, max(i, j))
        push!(cols, min(i, j))
    end
    return nothing
end

"""
    colloc_kernel_model(name, sysfor, Val(NX), Val(NU); T, backend,
                        element_hessian, ref_param_rounding, kwargs...) -> ExaModel

Build a `COLLOC`-family problem as an `ExaModel` whose `Ktyp` rows are filled
by KernelAbstractions kernels over [`colloc`](@ref), with the per-element
interval data resident on `backend`.

`backend = nothing` builds a host model that runs the same kernels on
KernelAbstractions' CPU backend; `backend = CUDABackend()` builds a device
model that never indexes a device array from the host, so it is usable with
`CUDA.allowscalar(false)`.

The non-external part -- the linear objective, the control-magnitude rows and
ORBIT2's terminal SQR/PROD rows -- is the same ordinary SIMD ExaModel
[`colloc_model`](@ref) builds, from the same `_colloc_simd_core`, and the
`Ktyp` elements go in as one `ExaModels.add_eval` with `f!`, `jac!`, `hess!`
and `hvp!`. `add_eval` rather than a `VectorNonlinearOracle` for the reason
given in COLLOC.jl: it fills rows that already exist, where a
`VectorNonlinearOracle` appends its own after every SIMD row and would permute
them relative to `sif_constraint_names`.

`element_hessian` and `ref_param_rounding` select between readings of the
Fortran exactly as in [`colloc_model`](@ref); any other `kwargs` go to
`ExaModelsSIF.sif_decode`. `prod = true` is the default because `hprod` on a
device needs it here; see the file header.

No kernel here uses `KernelAbstractions.@atomic`, so every callback is bitwise
reproducible across backends and `T` may be a non-bits type -- on
`backend = nothing`, which is the only backend a `BigFloat` can live on.

The declared Jacobian and Hessian patterns are identical to
[`colloc_model`](@ref)'s, entry for entry and in the same order, so the two
models' coordinate values are comparable slot by slot. The Hessian is blockwise
WITH duplicates across elements, so `nnzh` exceeds the number of distinct
nonzeros -- 265736 against 50964 for CAR2, 244899 against 32320 for ORBIT2 --
and the duplicates are summed by whoever assembles the matrix. Those distinct
counts are what CUTEst reports as `nnzh`, so the two agree on which entries can
be nonzero; what CUTEst does NOT supply is their VALUES on the COLLOC elements,
its `H` section being absent (see COLLOC.jl, note 1), which is why only the
host port can adjudicate a Hessian here.
"""
function colloc_kernel_model(name::AbstractString, sysfor, ::Val{NX}, ::Val{NU};
                             T = Float64, backend = nothing,
                             element_hessian::Bool = true,
                             ref_param_rounding::Bool = false,
                             prod::Bool = true,
                             kwargs...) where {NX,NU}
    D = _colloc_decode(name, sysfor, Val(NX), Val(NU); T, ref_param_rounding,
                       kwargs...)
    (; elems, nv, nvar, ncon, sys) = D
    _colloc_check_indices(elems, nvar, nv)
    _colloc_check_rows(elems, ncon)
    d = _colloc_kernel_data(elems, T, nvar, nv, Val(NX), backend)
    ka = backend === nothing ? KernelAbstractions.CPU() : backend
    nel = d.nel
    nhb = nv * (nv + 1) ÷ 2
    shapes = ntuple(id -> CollocShape{NX,NU,id}(), Val(NX))

    scratch = ExaModels.convert_array(zeros(T, nv * nel), backend)

    # Each callback declares one-based indexing before it gathers: the decode's
    # indices are absolute positions in `1:nvar`, so an offset array would be
    # read at the wrong variables silently.
    f! = (res, xv) -> begin
        Base.require_one_based_indexing(res, xv)
        fill!(res, zero(eltype(res)))
        ntuple(Val(NX)) do id
            p = d.pos[id]
            isempty(p) || _colloc_cons_kernel!(ka)(
                res, xv, p, d.idx, d.row, d.t, d.t1, d.w, sys, shapes[id];
                ndrange = length(p))
            nothing
        end
        KernelAbstractions.synchronize(ka)
        return nothing
    end
    jac! = (vals, xv) -> begin
        Base.require_one_based_indexing(vals, xv)
        ntuple(Val(NX)) do id
            p = d.pos[id]
            isempty(p) || _colloc_jac_kernel!(ka)(
                vals, xv, p, d.idx, d.t, d.t1, d.w, sys, shapes[id];
                ndrange = length(p))
            nothing
        end
        KernelAbstractions.synchronize(ka)
        return nothing
    end
    hess! = (vals, xv, yv) -> begin
        Base.require_one_based_indexing(vals, xv, yv)
        ntuple(Val(NX)) do id
            p = d.pos[id]
            isempty(p) || _colloc_hess_kernel!(ka)(
                vals, xv, p, d.idx, d.row, d.t, d.t1, d.w, yv, sys, shapes[id];
                ndrange = (nv, length(p)))
            nothing
        end
        KernelAbstractions.synchronize(ka)
        return nothing
    end
    hvp! = (Hv, xv, yv, vv) -> begin
        Base.require_one_based_indexing(Hv, xv, yv, vv)
        ntuple(Val(NX)) do id
            p = d.pos[id]
            isempty(p) || _colloc_hvpblock_kernel!(ka)(
                scratch, xv, vv, p, d.idx, d.row, d.t, d.t1, d.w, yv, sys,
                shapes[id]; ndrange = length(p))
            nothing
        end
        KernelAbstractions.synchronize(ka)
        _colloc_gather_kernel!(ka)(Hv, scratch, d.vptr, d.vsrc; ndrange = nvar)
        KernelAbstractions.synchronize(ka)
        return nothing
    end

    c, _x, con = _colloc_simd_core(T, backend, D)
    c, _ = ExaModels.add_eval(c, (con,), (_x,), f!;
        jac! = jac!,
        hess! = element_hessian ? hess! : nothing,
        hvp! = element_hessian ? hvp! : nothing,
        nnzj = nv * nel,
        nnzh = element_hessian ? nhb * nel : 0,
        jac_structure! = (rows, cols) ->
            _colloc_jac_structure!(rows, cols, elems, nv),
        hess_structure! = element_hessian ?
            ((rows, cols) -> _colloc_hess_structure!(rows, cols, elems, nv)) :
            nothing,
        adapt = Val(false))
    return ExaModels.ExaModel(c; prod = prod)
end

"""
    car2_kernel_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

CAR2 as an `ExaModel` whose external `COLLOC` rows are filled by
KernelAbstractions kernels over [`colloc`](@ref) and `CarODE`. The
kernel-backed counterpart of [`car2_model`](@ref); keywords are
[`colloc_kernel_model`](@ref)'s.
"""
car2_kernel_model(; T = Float64, backend = nothing, kwargs...) =
    colloc_kernel_model("CAR2", cf -> CarODE(T(only(unique(cf)))), Val(4),
                        Val(2); T, backend, kwargs...)

"""
    orbit2_kernel_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

ORBIT2 as an `ExaModel` whose external `COLLOC` rows are filled by
KernelAbstractions kernels over [`colloc`](@ref) and [`orbit2_ode`](@ref). The
kernel-backed counterpart of [`orbit2_model`](@ref); keywords are
[`colloc_kernel_model`](@ref)'s.
"""
orbit2_kernel_model(; T = Float64, backend = nothing, kwargs...) =
    colloc_kernel_model("ORBIT2", _ -> orbit2_ode(T), Val(6), Val(3);
                        T, backend, kwargs...)
