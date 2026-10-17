# JIMACK as an ExaModel whose objective callbacks are KernelAbstractions
# kernels over `jimack_pointwise`, with the per-element geometry resident on
# the model's backend.
#
# `jimack_model` (JIMACK.jl) interfaces the same `jimack_pointwise` through
# host loops. This file's kernels let the model be built and evaluated with
# scalar indexing of device arrays disabled (`CUDA.allowscalar(false)`), which
# host loops over a device array cannot be.
#
# WHY A KERNEL AND NOT AN ExaModels EXPRESSION. Unchanged from JIMACK.jl: one
# element value is ~160 floating-point operations and ExaModels nests a type
# parameter per operation. The generic function is the port; this file is only
# the launch wiring.
#
# WHAT IS PARALLEL. One finite element per thread for the value, the gradient
# and the Hessian-vector product: 864 elements over 3549 variables, each
# element reading exactly the 24 nodal values of its 8 vertices. The Hessian
# runs one thread per (element, column) pair instead -- 24 x 864 = 20736
# threads -- because 864 threads is a dozen warps and leaves a device idle,
# while the columns of an element block are independent.
#
# DERIVATIVES. ForwardDiff inside the kernel, over `jimack_element_value`, with
# exactly the sweeps the host port uses: the gradient is one 24-partial level,
# and a Hessian-vector product is a 24-partial level over 1-partial duals, 48
# scalars per intermediate value. A full `ForwardDiff.hessian` of a
# 24-variable function would carry 625 components per value, and the elemental
# Hessian is assembled from 24 of the 48-scalar sweeps instead -- 24 passes
# over a ~160-operation function, against one pass that spills thirteen times
# as much per value.
#
# `Dual` is an immutable over an `NTuple`, so for a fixed 24 inputs it is
# stack-allocated and GPU-legal. What is NOT GPU-legal is accumulating into a
# variable captured by an inner closure: capturing a reassigned variable boxes
# it on the heap, and a heap allocation inside a kernel is an
# `InvalidIRError`. Every reduction here is either an unrolled `ntuple` write
# or the inlined recursion in JIMACK.jl's `_jimack_du`.
#
# SCATTER. Adjacent elements share the nodes of their common faces, so the
# gradient and the Hessian-vector product scatter-accumulate with atomics; the
# summation order is whatever the device schedules. The Hessian COORDINATE
# callback does not: each element owns a disjoint 300-slot slice of the value
# array, so that kernel is a pure write and is bitwise reproducible.
#
# HESSIAN PATTERN: BLOCKWISE WITH DUPLICATES, 300 x 864 = 259200 entries, the
# lower triangle of each element's own 24x24 block. Elements sharing a node
# repeat a (row, col) pair, and that repetition is how an entry accumulates
# every element's contribution, since each `hess!` call writes one value per
# declared entry and the consumer sums duplicates. The alternative -- the
# deduplicated union, 118824 entries, which is what `jimack_model` declares and
# what CUTEst reports as `nnzh` -- would need a 259200-entry slot table and an
# atomic read-modify-write per contribution, which would put this callback's
# summation order at the mercy of the scheduler too. Disjoint writes keep the
# coordinate path bitwise reproducible across backends, and that is the path
# with no first-order cross-check: `hprod` reads `hvp!` and never reads a
# coordinate at all, so a defect here is invisible to it.
#
# Requires `hess!` on ExaModels' `ScalarNonlinearOracle` (ExaModels
# `sif-integration` 555b4ed9 or later). Without it `hess_coord` has no device
# path, and with a pattern that repeats coordinates it is not merely slow but
# wrong: the column-by-column reconstruction writes the whole assembled entry
# into every duplicate slot and the consumer then sums them.
#
# ONE BUILDER FOR BOTH BACKENDS. `backend = nothing` runs the same kernels on
# KernelAbstractions' CPU backend. A CPU-versus-GPU comparison therefore tests
# the device rather than two unrelated implementations;
# `test/jimack_gpu_verify.jl` is that comparison.

using KernelAbstractions

# ---------------------------------------------------------------------------
# Device-resident element geometry.

"""
    JimackKernelData

The variable-independent part of all of JIMACK's `INTEGRAL` elements as a
struct of arrays on whatever device the model is built for, holding exactly
the fields of a [`JimackElement`](@ref): `idx` is 24 x nelem global variable
indices, `g1`/`g2` are 24 x nelem shape-function gradients at the two distinct
quadrature points, and `c1`/`c2` are 3 x nelem taper constants.

An element's 24 entries are contiguous in the first dimension, which is the
order a thread reads them in.
"""
struct JimackKernelData{MI, MT}
    idx::MI
    g1::MT
    g2::MT
    c1::MT
    c2::MT
    nelem::Int
end

"""
    _jimack_check_indices(els, nvar)

Check that every element maps each of its 24 elemental variables into
`1:nvar`.

The kernels gather `x[idx[t, i]]`, so this is where an index that the decode
got wrong is caught. It runs once per build, O(nelem), against callbacks that
cost milliseconds, and it reports the element and the elemental variable; the
same fault inside a kernel surfaces as an aborted launch with nothing pointing
back at the decode.
"""
function _jimack_check_indices(els, nvar::Integer)
    for (i, e) in pairs(els)
        for t in 1:24
            j = e.idx[t]
            1 <= j <= nvar || error(
                "JIMACK: element $i maps elemental variable $(JIMACK_EVARS[t]) " *
                "to index $j, outside 1:$nvar")
        end
    end
    return nothing
end

function _jimack_kernel_data(els::Vector{JimackElement{T}}, backend) where {T}
    nelem = length(els)
    idxh = Matrix{Int32}(undef, 24, nelem)
    g1h = Matrix{T}(undef, 24, nelem)
    g2h = Matrix{T}(undef, 24, nelem)
    c1h = Matrix{T}(undef, 3, nelem)
    c2h = Matrix{T}(undef, 3, nelem)
    for (i, e) in pairs(els)
        for t in 1:24
            idxh[t, i] = e.idx[t]
            g1h[t, i] = e.g1[t]
            g2h[t, i] = e.g2[t]
        end
        for t in 1:3
            c1h[t, i] = e.c1[t]
            c2h[t, i] = e.c2[t]
        end
    end
    cv = v -> ExaModels.convert_array(v, backend)
    return JimackKernelData(cv(idxh), cv(g1h), cv(g2h), cv(c1h), cv(c2h), nelem)
end

# ---------------------------------------------------------------------------
# Kernels. One `JimackElement` is reconstituted per thread from the struct of
# arrays, so the kernels call JIMACK.jl's `jimack_element_value` /
# `jimack_element_grad` / `jimack_element_hvp` unchanged -- the value, its
# operation order and its literals have one definition, not two.
#
# Every `idx` entry lands in `1:length(x)`, which the gathers below rely on:
# `_jimack_check_indices` establishes it once at build time, where it can name
# the element and the elemental variable that went wrong.

@inline function _jimack_element(idx, g1, g2, c1, c2, i)
    T = eltype(g1)
    return JimackElement{T}(
        ntuple(t -> Int(idx[t, i]), Val(24)),
        ntuple(t -> g1[t, i], Val(24)),
        ntuple(t -> g2[t, i], Val(24)),
        ntuple(t -> c1[t, i], Val(3)),
        ntuple(t -> c2[t, i], Val(3)))
end

@kernel function _jimack_val_kernel!(
        buf, @Const(x), @Const(idx), @Const(g1), @Const(g2), @Const(c1), @Const(c2))
    i = @index(Global)
    e = _jimack_element(idx, g1, g2, c1, c2, i)
    E = eltype(buf)
    buf[i] = jimack_element_value(e, ntuple(t -> E(x[e.idx[t]]), Val(24)))
end

@kernel function _jimack_grad_kernel!(
        gv, @Const(x), @Const(idx), @Const(g1), @Const(g2), @Const(c1), @Const(c2))
    i = @index(Global)
    e = _jimack_element(idx, g1, g2, c1, c2, i)
    E = eltype(gv)
    gp = jimack_element_grad(e, ntuple(t -> E(x[e.idx[t]]), Val(24)))
    ntuple(Val(24)) do t
        KernelAbstractions.@atomic gv[e.idx[t]] += gp[t]
        nothing
    end
end

@kernel function _jimack_hvp_kernel!(
        Hv, @Const(x), @Const(vv), @Const(idx), @Const(g1), @Const(g2),
        @Const(c1), @Const(c2))
    i = @index(Global)
    e = _jimack_element(idx, g1, g2, c1, c2, i)
    E = eltype(Hv)
    h = jimack_element_hvp(e, ntuple(t -> E(x[e.idx[t]]), Val(24)),
                           ntuple(t -> E(vv[e.idx[t]]), Val(24)))
    ntuple(Val(24)) do t
        KernelAbstractions.@atomic Hv[e.idx[t]] += h[t]
        nothing
    end
end

# One column of one element's 24x24 Hessian per thread, as the Hessian-vector
# product in the direction of that column. Slot `(r - 1) * r / 2 + j` is the
# (r, j) entry of the block's row-major lower triangle, `r >= j`, which is the
# order `jimack_kernel_model` declares. The group scale divides each element's
# contribution rather than the assembled entry: division distributes over the
# consumer's sum of duplicates, and doing it here saves a second pass over
# 259200 values.
@kernel function _jimack_hess_kernel!(
        vals, @Const(x), @Const(idx), @Const(g1), @Const(g2), @Const(c1),
        @Const(c2), scale)
    j, i = @index(Global, NTuple)
    e = _jimack_element(idx, g1, g2, c1, c2, i)
    E = eltype(vals)
    a = ntuple(t -> E(x[e.idx[t]]), Val(24))
    h = jimack_element_hvp(e, a, ntuple(t -> ifelse(t == j, one(E), zero(E)), Val(24)))
    base = (i - 1) * 300
    for r in j:24
        vals[base + ((r - 1) * r) ÷ 2 + j] = h[r] / scale
    end
end

# ---------------------------------------------------------------------------
# Assembly

"""
    _jimack_setup(T, fortran_literals) -> (P, els, nvar, scale)

Decode JIMACK, check that it has the shape both builders assume, and build one
[`JimackElement`](@ref) per `INTEGRAL` element in the requested reading of the
Fortran's bare GLOBALS literals.

`fortran_literals` is `:single` (what gfortran compiles, and the default) or
`:double`; see [`jimack_model`](@ref) and the header of `JIMACK.jl`.
"""
function _jimack_setup(::Type{T}, fortran_literals::Symbol) where {T}
    P = ExaModelsSIF.sif_decode(sif_problem_path("JIMACK"); external = :defer)
    P.external_types == Dict("INTEGRAL" => :IEINT) || error(
        "JIMACK: expected a single deferred INTEGRAL element type calling " *
        "IEINT, found $(P.external_types)")
    all(==(1.0), P.varscale) || error("JIMACK: variable scaling is not supported here")
    length(P.grouporder) == 1 ||
        error("JIMACK: expected 1 group, found $(length(P.grouporder))")
    g = P.groups[only(P.grouporder)]
    (g.kind == 'N' && isempty(g.lin) && g.constant == 0.0 && g.range_ === nothing &&
     isempty(g.gparvals)) || error("JIMACK: unexpected objective group shape")
    all(==(1.0), last.(g.elems)) ||
        error("JIMACK: objective group weights its elements by " *
              "$(unique(last.(g.elems))); expected all 1")
    elemnames = first.(g.elems)

    # The grid, from the elements rather than from the Fortran's PARAMETER
    # statement, so a re-parameterized instance is caught instead of silently
    # mis-meshed. `(I,J,K)` runs over 0:N-1 x 0:N-1 x 0:M-1.
    ijk = [(trunc(Int, P.elements[en].parvals[:RI]),
            trunc(Int, P.elements[en].parvals[:RJ]),
            trunc(Int, P.elements[en].parvals[:RK])) for en in elemnames]
    n = maximum(first, ijk) + 1
    nj = maximum(t -> t[2], ijk) + 1
    m = maximum(last, ijk) + 1
    n == nj || error("JIMACK: the grid is $n x $nj in x,y; the Fortran assumes N x N")
    nvar = length(P.varnames)
    (length(elemnames) == n * n * m && nvar == 3 * (n + 1)^2 * (m + 1)) || error(
        "JIMACK: $(length(elemnames)) elements and $nvar variables do not match " *
        "an $(n)x$(n)x$(m) grid")

    # DELTA appears twice: as a proper double in the DATA sections (where it
    # sets the group scale) and as a bare literal in the Fortran GLOBALS. The
    # SIF says in both places that they must agree; check it, then round the
    # Fortran's copy.
    delta_data = 2 * m * n^2 / g.scale
    isapprox(delta_data, 0.1; rtol = 1e-12) || error(
        "JIMACK: the group scale implies DELTA = $delta_data, but the appended " *
        "Fortran hard-codes 0.1; the SIF requires the two to agree")

    lit = if fortran_literals === :single
        # what gfortran does with a bare real literal
        v -> T(Float64(Float32(v)))
    elseif fortran_literals === :double
        v -> T(v)
    else
        error("JIMACK: fortran_literals must be :single or :double, got " *
              "$(repr(fortran_literals))")
    end

    els = jimack_elements(P, T, elemnames; n, m,
                          delta = lit(0.1), epsv = lit(0.01), w = lit(0.577350269189))
    _jimack_check_indices(els, nvar)
    return P, els, nvar, T(g.scale)
end

"""
    jimack_kernel_model(; T = Float64, backend = nothing,
                          fortran_literals = :single) -> ExaModel

Build JIMACK as an `ExaModel` whose objective callbacks are
KernelAbstractions kernels over [`jimack_pointwise`](@ref), with the
per-element geometry resident on `backend`.

`backend = nothing` builds a host model that runs the same kernels on
KernelAbstractions' CPU backend; `backend = CUDABackend()` builds a device
model that never indexes a device array from the host, so it is usable with
`CUDA.allowscalar(false)`.

Unconstrained, 3549 variables, no bounds; the objective is the sum of the 864
element energies divided by the SIF group scale `2*M*N^2/DELTA = 17280`, and
goes in as a single `ScalarNonlinearOracle` with `f`, `grad!`, `hvp!` and an
explicit `hess!`. `fortran_literals` selects the reading of the Fortran's bare
GLOBALS constants exactly as in [`jimack_model`](@ref), and `:single` -- the
default -- is the one that reproduces CUTEst's objective bit for bit.

`T` must be a bits type: the gradient and the Hessian-vector product
scatter-accumulate with `KernelAbstractions.@atomic`, which has no
implementation for a `BigFloat`, and that holds for `backend = nothing` too
because the kernels run through KA's CPU backend either way.
[`jimack_model`](@ref) serves a non-bits element type.

The declared Hessian sparsity is the lower triangle of each element's own
24x24 block, 300 x 864 = 259200 entries WITH duplicates across elements that
share a node -- not the 118824-entry deduplicated union that
[`jimack_model`](@ref) declares and that CUTEst reports as `nnzh`. Both
assemble to the same matrix, since coordinate duplicates are summed; the
blockwise pattern is what lets `hess!` be a pure disjoint write. Compare the
two models' Hessians as assembled matrices, not entry by entry.

!!! note "What the device buys, and what it does not"
    Measured per call on shin-compute-001 (Quadro GV100, one Julia thread,
    minimum of twenty calls; `hess_coord!` of five):

    | callback | kernels, host | kernels, CUDA | `jimack_model` |
    |---|---|---|---|
    | `obj` | 0.147 ms | 0.139 ms | 0.103 ms |
    | `grad!` | 1.28 ms | 0.91 ms | 1.07 ms |
    | `hprod!` | 4.59 ms | 2.06 ms | 4.30 ms |
    | `hess_coord!` | 106 ms | 11.2 ms | 16.1 s |

    864 elements is fourteen warps, far too few to fill this device, so `obj`,
    `grad!` and `hprod!` gain almost nothing from it -- the problem is too
    small, not the kernels too slow. `hess_coord!` is the exception because it
    launches one thread per (element, column), 20736 of them, and because the
    comparison on the right is against reconstructing the Hessian from one
    `hvp!` per variable: 3549 host passes over all 864 elements.
"""
function jimack_kernel_model(; T = Float64, backend = nothing,
                             fortran_literals::Symbol = :single)
    P, els, nvar, scale = _jimack_setup(T, fortran_literals)
    d = _jimack_kernel_data(els, backend)
    ka = backend === nothing ? KernelAbstractions.CPU() : backend
    nelem = d.nelem

    # Hessian sparsity: the lower triangle of each element's 24x24 block, in
    # the slot order `_jimack_hess_kernel!` writes. See the docstring and the
    # header for why the duplicates are intended.
    hr = Vector{Int}(undef, 300 * nelem)
    hc = Vector{Int}(undef, 300 * nelem)
    let p = 0
        for e in els, t1 in 1:24, t2 in 1:t1
            p += 1
            i, j = e.idx[t1], e.idx[t2]
            hr[p], hc[p] = max(i, j), min(i, j)
        end
    end

    # The decode's indices are absolute positions in `1:nvar`, so a gather
    # against an offset array would read the wrong variables silently.
    valbuf = ExaModels.convert_array(zeros(T, nelem), backend)
    f = xv -> begin
        Base.require_one_based_indexing(xv)
        _jimack_val_kernel!(ka)(valbuf, xv, d.idx, d.g1, d.g2, d.c1, d.c2;
                                ndrange = nelem)
        KernelAbstractions.synchronize(ka)
        return sum(valbuf) / scale
    end
    grad! = (gv, xv) -> begin
        Base.require_one_based_indexing(gv, xv)
        fill!(gv, zero(eltype(gv)))
        _jimack_grad_kernel!(ka)(gv, xv, d.idx, d.g1, d.g2, d.c1, d.c2;
                                 ndrange = nelem)
        KernelAbstractions.synchronize(ka)
        gv ./= eltype(gv)(scale)
        return gv
    end
    hvp! = (Hv, xv, vv) -> begin
        Base.require_one_based_indexing(Hv, xv, vv)
        fill!(Hv, zero(eltype(Hv)))
        _jimack_hvp_kernel!(ka)(Hv, xv, vv, d.idx, d.g1, d.g2, d.c1, d.c2;
                                ndrange = nelem)
        KernelAbstractions.synchronize(ka)
        Hv ./= eltype(Hv)(scale)
        return Hv
    end
    hess! = (vals, xv) -> begin
        Base.require_one_based_indexing(vals, xv)
        _jimack_hess_kernel!(ka)(vals, xv, d.idx, d.g1, d.g2, d.c1, d.c2,
                                 eltype(vals)(scale); ndrange = (24, nelem))
        KernelAbstractions.synchronize(ka)
        return vals
    end

    c = ExaModels.ExaCore(T; backend = backend)
    c, _ = ExaModels.add_var(c, nvar; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    c = ExaModels.objective(c, ExaModels.ScalarNonlinearOracle(;
        nvar = nvar, f = f, grad! = grad!, hvp! = hvp!, hess! = hess!,
        hess_rows = hr, hess_cols = hc, adapt = Val(false)))
    return ExaModels.ExaModel(c)
end
