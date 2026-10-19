# SARO / SAROMM as ExaModels whose constraint callbacks are
# KernelAbstractions kernels over `saro_stage_values`, with the per-stage
# driving data and index tables resident on the model's backend.
#
# `saro_oracle_model` (SARO.jl) interfaces the same `saro_stage_values` through
# host loops. This file's kernels let the models be built and evaluated with
# scalar indexing of device arrays disabled (`CUDA.allowscalar(false)`), which
# host loops over a device array cannot be. The value, its operation order and
# its literals have one definition -- SARO.jl's -- not two; only the AD seeding
# and the launch wiring are here.
#
# WHY A KERNEL AND NOT AN ExaModels EXPRESSION. Unchanged from SARO.jl: one
# stage is two RK4 steps, eight `saro_rhs` evaluations, ~500 elementary
# operations, and ExaModels nests a type parameter per operation.
#
# WHAT IS PARALLEL. The 365 stages are independent -- stage `k` reads
# `X(1..9,k), U(1,k), U(2,k)` and no variable of any other stage -- so there is
# no coupling to serialize on. The difficulty is register pressure, not
# parallelism, and it is what fixes the thread shape of every kernel here.
#
# THE SHAPE IS CHOSEN BY MEASURED REGISTER PRESSURE, not by analogy with the
# other ports. A GV100 gives a thread 255 registers; past that the compiler
# spills to local memory, which is device memory. `JIMACK_gpu.jl` assembles an
# element Hessian from one thread per (element, COLUMN), each a single-partial
# sweep over a 24-partial sweep -- 48 Float64 per intermediate value. Here that
# shape fits but spills hard, so each kernel instead carries the NARROWEST
# derivative that still produces one output, and the thread count rises to
# compensate. Measured on a GV100 over all 365 stages, Float64, `@cuda` with
# 256 threads per block, minimum of five launches:
#
#   | kernel            | per thread    | threads | regs | spill/thr | compile | time     |
#   |-------------------|---------------|---------|------|-----------|---------|----------|
#   | jac,   1 / stage  | 11-partial    |     365 |  255 |  18680 B  |  43.2 s | 0.417 ms |
#   | jac,  11 / stage  | 1-partial     |    4015 |  185 |   2672 B  |   0.9 s | 0.050 ms |
#   | hess, 11 / stage  | 11-p over 1-p |    4015 |  255 |  86384 B  |   6.2 s | 3.058 ms |
#   | hess, 66 / stage  | 1-p over 1-p  |   24090 |  255 |   5480 B  |   1.8 s | 0.411 ms |
#   | hvp,  11 / stage  | 1-p over 1-p  |    4015 |  255 |   5480 B  |   1.0 s | 0.087 ms |
#   | hess,  1 / stage  | 11-p over 11-p|     365 |  255 | 520456 B  |  39.3 s | not run  |
#
# The last row is the shape a plain `ForwardDiff.hessian` in a kernel would
# take, and it is the one to rule out first: twice-nested 11-partial duals are
# 144 Float64 per intermediate value, and the compiled kernel asks for 508 KB
# of local memory per thread -- at the 512 KB a thread may have on compute
# capability 7.0. It is not timed here because the shape is excluded by that
# number, not by a measurement of it.
#
# The narrow shape evaluates the ~500-operation stage function six times as
# often for the Hessian and eleven times as often for the Jacobian, and is
# still seven times faster for the Hessian and eight for the Jacobian, because
# the wide shape's spill traffic costs more than the redundant arithmetic. It
# also compiles in a fraction of the time, and the Jacobian's wide shape is the
# extreme case: 43 s to compile 365 threads' worth of 11-partial duals. So:
#
#   * `f!`     one thread per STAGE -- a plain value, no duals, nothing to
#              spill, and one evaluation already yields all eleven outputs.
#   * `jac!`   one thread per (stage, elemental VARIABLE): a single-partial
#              sweep gives that variable's column, all eleven slots of it.
#   * `hess!`  one thread per (stage, lower-triangle PAIR): two single-partial
#              levels give exactly one second mixed partial.
#   * `hvp!`   one thread per (stage, elemental variable): the same two
#              single-partial levels with the second direction taken from `v`
#              instead of a unit vector.
#
# `saro_rhs` stays `@noinline`, as on the host. Forcing the eight copies inline
# on the device moves nothing worth having: in the (stage, column) Hessian
# shape it took compilation from 7.0 s to 34.0 s and left the spill at 79000 B
# against 86440 B, both an order of magnitude above what the narrow shape
# needs.
#
# NO ATOMICS ANYWHERE, so every callback is bitwise reproducible across
# backends and the builder serves a non-bits element type.
#
#   * `f!` is two kernels. The first writes every row's constant and linear
#     terms, one thread per row, summing in the order the decode lists them.
#     The second adds each stage's eleven element values, one thread per stage,
#     and that is a disjoint read-modify-write because a row holds at most one
#     element -- `_saro_check_rows` establishes it at build time.
#   * `jac!` and `hess!` are disjoint writes: each stage owns a contiguous
#     slice of the value array.
#   * `hvp!` is the only accumulation, and it runs as two kernels rather than
#     one atomic one: the first writes each stage's eleven contributions into
#     its own slice of a scratch array, the second sums, one thread per
#     VARIABLE, over the contributions listed for it in a table built once at
#     build time. The summation order is that table's and so is the same on
#     every backend. The stages happen not to share variables at all, which
#     would make the gather a permutation; it is kept because correctness then
#     does not depend on that.
#
# WHY `hvp!` IS SUPPLIED. Without it ExaModels' `hprod!` accumulates `hess!`'s
# 24090 coordinates on the HOST, which on a device model means copying the
# value array back every call. Supplying it also makes `hprod` independent of
# the declared sparsity, which is what lets the two be compared against each
# other (`test/saro_gpu_verify.jl`).
#
# HESSIAN PATTERN: the lower triangle of each stage's own 11x11 block, 66 x 365
# = 24090 entries, in the slot order the kernel writes -- entry for entry and
# in the same order as [`saro_oracle_model`](@ref). So the two builders' values
# are comparable slot by slot rather than only as assembled matrices, which is
# the stronger reading: it sees a fill that lands a stage's values in the wrong
# slots of its own block. The pattern is correct only for a stage whose
# variable map is injective, which `_saro_check_indices` requires rather than
# assumes -- the defect `RAYBENDS_gpu.jl` records, where an elemental pair
# whose two slots are the same problem variable must be counted twice. SARO's
# stages are injective, and across stages there are no duplicate coordinates
# either, so here the blockwise pattern and the deduplicated union coincide and
# `nnzh` is 24090 on both builders.
#
# WHY THE MODEL IS BUILT WITH `prod = true`. The objective is an ordinary SIMD
# ExaModels expression (it is linear in both problems) and the constraints are
# an oracle. ExaModels' KernelAbstractions override of `hprod!` for a model
# with oracles dispatches on the prodhelper that `prod = true` builds, and
# without it the call falls back to the generic method, which evaluates the
# SIMD part through host loops over the device arrays. `COLLOC_gpu.jl` and
# `src/native/HS88_92.jl` carry the same note for the same reason.
#
# FLOAT32 DERIVATIVES ARE FINITE because `saro_rhs` widens the one quotient
# whose denominator is the regulariser alone (`_saro_quot` in SARO.jl); the
# kernels inherit it. For Float64 the widening folds away at compile time, so
# the Float64 kernels' code, registers and spill are those of the plain
# division.
#
# ONE BUILDER FOR BOTH BACKENDS. `backend = nothing` runs the same kernels on
# KernelAbstractions' CPU backend. A CPU-versus-GPU comparison therefore tests
# the device rather than two unrelated implementations;
# `test/saro_gpu_verify.jl` is that comparison.

using KernelAbstractions

# ---------------------------------------------------------------------------
# The narrow derivative sweeps.
#
# ForwardDiff is seeded by hand rather than through `ForwardDiff.jacobian` /
# `ForwardDiff.hessian`: those take an `AbstractVector`, allocate a result and
# consult a mutable tag counter, none of which is available in a kernel.
# Seeding by hand also keeps the partial count at the minimum each output
# needs, which is the whole point here.

"""
    _saro_jac_col(da, db, v, j) -> NTuple{11}

Column `j` of a stage's 11x11 Jacobian: the derivative of all eleven element
values with respect to the stage's `j`-th elemental variable, by one
single-partial forward sweep over [`saro_stage_values`](@ref).

Each partial of a wider sweep is computed from the same formulas as the single
partial of this one, so this reproduces column `j` of
[`saro_stage_jac`](@ref) bit for bit.
"""
@inline function _saro_jac_col(da::NTuple{11,S}, db::NTuple{11,S},
                               v::NTuple{11,T}, j) where {S,T}
    D = ForwardDiff.Dual{:SAROcol,T,1}
    vd = ntuple(i -> D(v[i],
                       ForwardDiff.Partials((ifelse(i == j, one(T), zero(T)),))),
                Val(11))
    out = saro_stage_values(da, db, vd)
    return ntuple(s -> ForwardDiff.partials(out[s], 1), Val(11))
end

"""
    _saro_wsum(w, out)

`sum(w[s] * out[s])` accumulated left to right, which is the order
[`saro_stage_hess`](@ref) uses. Written out rather than reduced, because the
association is what the Hessian's last bits depend on and because a reduction
over a closure that updates a captured accumulator boxes it on the heap, which
inside a kernel is an `InvalidIRError`.
"""
@inline _saro_wsum(w::NTuple{11,Any}, o::NTuple{11,Any}) =
    ((((((((((w[1] * o[1] + w[2] * o[2]) + w[3] * o[3]) + w[4] * o[4]) +
            w[5] * o[5]) + w[6] * o[6]) + w[7] * o[7]) + w[8] * o[8]) +
        w[9] * o[9]) + w[10] * o[10]) + w[11] * o[11])

"""
    _saro_h2(da, db, v, w, d1, d2) -> scalar

The second directional derivative `d1' * H * d2` of
`sum(w[s] * saro_stage_values(da, db, v)[s])`, as two nested single-partial
sweeps -- four scalars per intermediate value.

With `d1 = e_a` and `d2 = e_b` this is the single entry `H[a, b]`, which is how
the Hessian coordinate kernel reads it; with `d2` taken from the direction
vector it is component `a` of the stage's Hessian-vector product.
"""
@inline function _saro_h2(da::NTuple{11,S}, db::NTuple{11,S}, v::NTuple{11,T},
                          w::NTuple{11,T}, d1::NTuple{11,T},
                          d2::NTuple{11,T}) where {S,T}
    O = ForwardDiff.Dual{:SAROd2,T,1}
    D = ForwardDiff.Dual{:SAROd1,O,1}
    vd = ntuple(i -> D(O(v[i], ForwardDiff.Partials((d2[i],))),
                       ForwardDiff.Partials((O(d1[i],
                           ForwardDiff.Partials((zero(T),))),))),
                Val(11))
    s = _saro_wsum(w, saro_stage_values(da, db, vd))
    return ForwardDiff.partials(ForwardDiff.partials(s, 1), 1)
end

# ---------------------------------------------------------------------------
# Device-resident stage data and index tables.

"""
    SAROKernelData

Everything the kernels read that does not depend on `x`, as a struct of arrays
on whatever device the model is built for.

`da`/`db` are 11 x nstage driving-data tuples -- the two halves of
[`saro_stage_data`](@ref) -- `idx` is 11 x nstage global variable indices in
[`SARO_EVARS`](@ref) order, and `rows` is 11 x nstage constraint rows, zero
where the SIF asks for no such element slot. `jpos[slot, i]` is the position in
the Jacobian value array of that slot's first of eleven entries, zero where the
slot is unused.

`rconst`/`lptr`/`lcol`/`lval` are the rows' constants and their linear terms in
compressed-row form, `pa`/`pb` enumerate the 66 lower-triangle pairs of a
stage's block in declaration order, and `vptr`/`vsrc` are the gather table the
Hessian-vector product sums through: `vsrc[p]` for `p in vptr[j]:vptr[j+1]-1`
are the flat positions in an 11 x nstage scratch array that contribute to
variable `j`.

A stage's eleven entries are contiguous in the first dimension of `da`, `db`,
`idx` and `rows`, which is the order a thread reads them in.
"""
struct SAROKernelData{MT,MI,VT,VI}
    da::MT
    db::MT
    idx::MI
    rows::MI
    jpos::MI
    rconst::VT
    lptr::VI
    lcol::VI
    lval::VT
    pa::VI
    pb::VI
    vptr::VI
    vsrc::VI
    nst::Int
end

"""
    _saro_check_indices(stages, nvar)

Check that every stage maps its eleven elemental variables to eleven DISTINCT
indices in `1:nvar`.

Two kernel properties rest on this. The gathers `x[idx[a, i]]` have no bounds
check of their own, and on a device an out-of-range gather has no useful
diagnosis -- it reads whatever memory is there. And the blockwise Hessian
pattern is only right for an injective map: the problem Hessian entry `(i, j)`
sums the stage Hessian over every ordered slot pair landing on it, so a slot
pair `a != b` whose two slots were the same variable would have to be counted
twice on that variable's diagonal, where the lower triangle visits it once.
The SIF declares `X(1..9,k)`, `U(1,k)` and `U(2,k)` as eleven distinct
variables; a SIF that identified two of them would need the Hessian assembled
differently and is refused rather than assembled wrongly.

It runs once per build, O(nstage), against callbacks that cost milliseconds,
and it names the stage and the elemental variable; the same fault inside a
kernel surfaces as an aborted launch, or as a silently wrong answer, with
nothing pointing back at the decode that produced the index.
"""
function _saro_check_indices(stages, nvar::Integer)
    for s in stages
        for a in 1:11
            j = s.idx[a]
            1 <= j <= nvar || error(
                "SARO: stage $(s.k) maps elemental variable $(SARO_EVARS[a]) " *
                "to index $j, outside 1:$nvar")
            for b in 1:(a-1)
                s.idx[b] == j && error(
                    "SARO: stage $(s.k) maps elemental variables " *
                    "$(SARO_EVARS[b]) and $(SARO_EVARS[a]) to the same " *
                    "variable index $j; the blockwise Hessian pattern assumes " *
                    "they are distinct")
            end
        end
    end
    return nothing
end

"""
    _saro_check_rows(stages, ncon)

Check that no constraint row receives an element value from more than one
(stage, slot) pair.

The value kernel ADDS `res[rows[slot, i]]`, one thread per stage, which is a
disjoint read-modify-write only under that condition. The SIF gives each
`F(i,k)` and `G(i,k)` its own group, so no row takes two; a SIF that put two
external elements in one group would need an accumulation here, and it is
better to be refused than to have one stage's value silently overwrite the
other's.
"""
function _saro_check_rows(stages, ncon::Integer)
    seen = falses(ncon)
    for s in stages, slot in 1:11
        r = s.rows[slot]
        r == 0 && continue
        1 <= r <= ncon ||
            error("SARO: stage $(s.k) slot $slot fills row $r, outside 1:$ncon")
        seen[r] && error(
            "SARO: row $r receives an element value from more than one stage " *
            "slot; the kernels assume at most one per row")
        seen[r] = true
    end
    return nothing
end

"""
    _saro_scatter_table(stages, nvar) -> (vptr, vsrc)

The variable-wise transpose of the stage-variable map: `vsrc[vptr[j]]` through
`vsrc[vptr[j+1]-1]` are the flat positions `(i-1)*11 + a` of every stage slot
that holds variable `j`.

This is what lets the Hessian-vector product accumulate without an atomic: the
scatter becomes a gather, one thread per variable, summing in the order this
table lists -- the same order on every backend. A variable no stage touches
gets the zero its empty list sums to.
"""
function _saro_scatter_table(stages, nvar::Integer)
    cnt = zeros(Int, nvar + 1)
    for s in stages, a in 1:11
        cnt[s.idx[a]+1] += 1
    end
    vptr = Vector{Int32}(undef, nvar + 1)
    vptr[1] = 1
    for j in 1:nvar
        vptr[j+1] = vptr[j] + cnt[j+1]
    end
    nxt = copy(vptr)
    vsrc = Vector{Int32}(undef, 11 * length(stages))
    for (i, s) in pairs(stages), a in 1:11
        j = s.idx[a]
        vsrc[nxt[j]] = (i - 1) * 11 + a
        nxt[j] += 1
    end
    return vptr, vsrc
end

# ---------------------------------------------------------------------------
# Kernels.
#
# Every `idx` entry lands in `1:length(x)`, every nonzero `rows` entry in
# `1:ncon` and at most once, which the gathers and the row update below rely
# on: `_saro_check_indices` and `_saro_check_rows` establish both once at build
# time, where they can name the stage and the slot that went wrong.

@inline _saro_gatherd(A, i) = ntuple(a -> A[a, i], Val(11))
@inline _saro_unit(::Type{T}, j) where {T} =
    ntuple(a -> ifelse(a == j, one(T), zero(T)), Val(11))

# Row constants and linear terms, one thread per row, summed in the order the
# decode lists them -- which is also the order `saro_oracle_model`'s `f!` adds
# them in, so the two agree to the last bit.
@kernel function _saro_lin_kernel!(res, @Const(x), @Const(rconst), @Const(lptr),
                                   @Const(lcol), @Const(lval))
    r = @index(Global)
    E = eltype(res)
    acc = E(rconst[r])
    for p in lptr[r]:(lptr[r+1]-1)
        acc += E(lval[p]) * x[lcol[p]]
    end
    res[r] = acc
end

# Each stage's eleven element values added to the rows that hold them. One
# evaluation of the ~500-operation stage function per thread serves all eleven.
@kernel function _saro_cons_kernel!(res, @Const(x), @Const(da), @Const(db),
                                    @Const(idx), @Const(rows))
    i = @index(Global)
    E = eltype(res)
    out = saro_stage_values(_saro_gatherd(da, i), _saro_gatherd(db, i),
                            ntuple(a -> E(x[idx[a, i]]), Val(11)))
    ntuple(Val(11)) do slot
        r = rows[slot, i]
        r == 0 || (res[r] += out[slot])
        nothing
    end
end

# One column of one stage's Jacobian block per thread: the eleven element
# values' derivatives with respect to one elemental variable. Slot `j` of the
# eleven entries `jpos[slot, i]` starts is the derivative in elemental variable
# `j`, which is the order `saro_oracle_model` declares.
@kernel function _saro_jac_kernel!(vals, @Const(x), @Const(da), @Const(db),
                                   @Const(idx), @Const(jpos))
    j, i = @index(Global, NTuple)
    E = eltype(vals)
    c = _saro_jac_col(_saro_gatherd(da, i), _saro_gatherd(db, i),
                      ntuple(a -> E(x[idx[a, i]]), Val(11)), j)
    ntuple(Val(11)) do slot
        p = jpos[slot, i]
        p == 0 || (vals[p+j-1] = c[slot])
        nothing
    end
end

# One lower-triangle entry of one stage's Hessian block per thread. Pair `q` is
# `(pa[q], pb[q])` with `pb[q] <= pa[q]`, enumerated in the order
# `saro_oracle_model` declares -- `for a in 1:11, b in 1:a` -- so slot
# `(i-1)*66 + q` of the value array is the same entry in both builders.
@kernel function _saro_hess_kernel!(vals, @Const(x), @Const(da), @Const(db),
                                    @Const(idx), @Const(rows), @Const(mult),
                                    @Const(pa), @Const(pb))
    q, i = @index(Global, NTuple)
    E = eltype(vals)
    w = ntuple(Val(11)) do slot
        r = rows[slot, i]
        r == 0 ? zero(E) : E(mult[r])
    end
    vals[(i-1)*66+q] = _saro_h2(
        _saro_gatherd(da, i), _saro_gatherd(db, i),
        ntuple(a -> E(x[idx[a, i]]), Val(11)), w,
        _saro_unit(E, pa[q]), _saro_unit(E, pb[q]))
end

# Hessian-vector product, stage one: one thread per (stage, elemental
# variable), writing that stage's contribution to that variable into its own
# slice of the scratch array -- a disjoint write.
@kernel function _saro_hvpblock_kernel!(sc, @Const(x), @Const(vv), @Const(da),
                                        @Const(db), @Const(idx), @Const(rows),
                                        @Const(mult))
    a, i = @index(Global, NTuple)
    E = eltype(sc)
    w = ntuple(Val(11)) do slot
        r = rows[slot, i]
        r == 0 ? zero(E) : E(mult[r])
    end
    sc[(i-1)*11+a] = _saro_h2(
        _saro_gatherd(da, i), _saro_gatherd(db, i),
        ntuple(k -> E(x[idx[k, i]]), Val(11)), w,
        _saro_unit(E, a), ntuple(k -> E(vv[idx[k, i]]), Val(11)))
end

# Stage two: one thread per VARIABLE, summing the scratch entries listed for it
# by `_saro_scatter_table`. Assigns, so it needs no preceding fill.
@kernel function _saro_gather_kernel!(out, @Const(sc), @Const(vptr), @Const(vsrc))
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
    _saro_kernel_setup(name, T) -> NamedTuple

Decode SARO or SAROMM, check that it has the shape the kernels assume, and
return everything that does not depend on `x`: the stages, the row bounds and
constants, the rows' linear terms in compressed-row form, and the declared
Jacobian and Hessian patterns.

The patterns are built in exactly the order [`saro_oracle_model`](@ref) builds
them -- the linear entries, then eleven per element slot for the Jacobian, and
the lower triangle of each stage's block for the Hessian -- so the two builders
declare the same entries in the same positions and their coordinate values are
comparable slot by slot.
"""
function _saro_kernel_setup(name::AbstractString, ::Type{T}) where {T}
    path = sif_problem_path(name)
    P = ExaModelsSIF.sif_decode(path; external = :defer)
    for (tname, fn) in P.external_types
        (tname in ("FTYPE", "GTYPE") && fn === :SAROFN) || error(
            "$name: expected deferred FTYPE/GTYPE calling SAROFN, found " *
            "$tname => $fn")
    end
    all(==(1.0), P.varscale) || error("$name: variable scaling is not supported here")

    tab = _saro_tables_cached(path)
    stages, connames, objterms = saro_stages(P)
    isempty(stages) && error("$name: no FTYPE/GTYPE elements found")
    nvar = length(P.varnames)
    ncon = length(connames)
    _saro_check_indices(stages, nvar)
    _saro_check_rows(stages, ncon)

    # Row bounds and linear parts, straight from the decode. CUTEst (and
    # ExaModelsSIF's builder) fold a group's CONSTANT into the row VALUE and
    # leave the bound at zero, so `cons(x)` is the residual.
    lcon = zeros(T, ncon)
    ucon = zeros(T, ncon)
    rconst = zeros(T, ncon)
    lptr = Vector{Int32}(undef, ncon + 1)
    lcol = Int32[]
    lval = T[]
    lptr[1] = 1
    for (i, gn) in enumerate(connames)
        g = P.groups[gn]
        g.scale == 1.0 || error("$name: row $gn is scaled by $(g.scale)")
        g.range_ === nothing || error("$name: row $gn has a range")
        rconst[i] = -T(g.constant)
        if g.kind == 'E'
            lcon[i] = zero(T); ucon[i] = zero(T)
        elseif g.kind == 'G'
            lcon[i] = zero(T); ucon[i] = T(Inf)
        elseif g.kind == 'L'
            lcon[i] = T(-Inf); ucon[i] = zero(T)
        else
            error("$name: row $gn has unexpected kind $(g.kind)")
        end
        for (j, a) in g.lin
            push!(lcol, j); push!(lval, T(a))
        end
        lptr[i+1] = length(lval) + 1
    end
    nlin = length(lval)

    nst = length(stages)
    # Jacobian pattern: the linear entries, then eleven per element row. The
    # row's own linear variable is X(i,k+1) or S(i,k), never one of the stage's
    # eleven, so no coordinate repeats; duplicates would be summed anyway.
    nj = nlin + 11 * sum(count(!=(0), s.rows) for s in stages)
    jr = Vector{Int}(undef, nj)
    jc = Vector{Int}(undef, nj)
    jpos = zeros(Int32, 11, nst)
    let p = 0
        for (i, gn) in enumerate(connames)
            for (j, _) in P.groups[gn].lin
                p += 1
                jr[p] = i
                jc[p] = j
            end
        end
        p == nlin || error("$name: linear Jacobian block is $p, expected $nlin")
        for (si, s) in pairs(stages), slot in 1:11
            r = s.rows[slot]
            r == 0 && continue
            jpos[slot, si] = p + 1
            for a in 1:11
                p += 1
                jr[p] = r
                jc[p] = s.idx[a]
            end
        end
    end

    # Hessian pattern: the lower triangle of each stage's 11x11 block.
    hr = Vector{Int}(undef, 66 * nst)
    hc = Vector{Int}(undef, 66 * nst)
    pa = Vector{Int32}(undef, 66)
    pb = Vector{Int32}(undef, 66)
    let p = 0, q = 0
        for a in 1:11, b in 1:a
            q += 1
            pa[q], pb[q] = a, b
        end
        for s in stages, q2 in 1:66
            p += 1
            i, j = s.idx[pa[q2]], s.idx[pb[q2]]
            hr[p], hc[p] = max(i, j), min(i, j)
        end
    end

    return (; P, tab, stages, connames, objterms, nvar, ncon, nst, nlin,
            lcon, ucon, rconst, lptr, lcol, lval, jr, jc, jpos, hr, hc, pa, pb)
end

function _saro_kernel_data(D, ::Type{T}, backend) where {T}
    nst = D.nst
    dah = Matrix{T}(undef, 11, nst)
    dbh = Matrix{T}(undef, 11, nst)
    idxh = Matrix{Int32}(undef, 11, nst)
    rowsh = Matrix{Int32}(undef, 11, nst)
    for (i, s) in pairs(D.stages)
        da, db = saro_stage_data(D.tab, s.k, T)
        for a in 1:11
            dah[a, i] = da[a]
            dbh[a, i] = db[a]
            idxh[a, i] = s.idx[a]
            rowsh[a, i] = s.rows[a]
        end
    end
    vptr, vsrc = _saro_scatter_table(D.stages, D.nvar)
    cv = v -> ExaModels.convert_array(v, backend)
    return SAROKernelData(cv(dah), cv(dbh), cv(idxh), cv(rowsh), cv(D.jpos),
                          cv(D.rconst), cv(D.lptr), cv(D.lcol), cv(D.lval),
                          cv(D.pa), cv(D.pb), cv(vptr), cv(vsrc), nst)
end

"""
    saro_kernel_model(name; T = Float64, backend = nothing, prod = true) -> ExaModel

Build SARO or SAROMM as an `ExaModel` whose constraint callbacks are
KernelAbstractions kernels over [`saro_stage_values`](@ref), with the per-stage
driving data resident on `backend`.

`backend = nothing` builds a host model that runs the same kernels on
KernelAbstractions' CPU backend; `backend = CUDABackend()` builds a device
model that never indexes a device array from the host, so it is usable with
`CUDA.allowscalar(false)`.

All rows go into one `VectorNonlinearOracle` in SIF group order (so row `i` is
`sif_constraint_names(name)[i]`), exactly as in [`saro_oracle_model`](@ref),
with `f!`, `jac!`, `hess!` and `hvp!`; the linear objective is an ordinary SIMD
ExaModels expression. The declared Jacobian and Hessian patterns are identical
to that builder's, entry for entry and in the same order, so the two models'
coordinate values are comparable slot by slot. `nnzh` is 24090 on both: the
lower triangle of each of the 365 stages' 11x11 blocks, with no coordinate
repeated, because no two stages share a variable.

No kernel uses `KernelAbstractions.@atomic`, so every callback is bitwise
reproducible across backends and `T` may be a non-bits type -- on
`backend = nothing`, which is the only backend a `BigFloat` can live on.

`prod = true` is the default because `hprod` needs it on a device here: the
objective is a SIMD expression and the constraints an oracle, and only the
prodhelper that `prod = true` builds gives that mixture a device `hprod` path.

In Float32 every callback is finite on every backend, for the reason
[`saro_oracle_model`](@ref) records: the one quotient in `saro_rhs` whose
denominator is the regulariser alone is differentiated over Float64. On the
host backend the Float32 `cons`, `jac` and `hess` are bit-identical to
[`saro_oracle_model`](@ref)'s.

!!! note "What the device buys"
    Measured per call on shin-compute-001 (Quadro GV100), one Julia thread,
    minimum of ten calls for the kernels and three for the host port:

    | callback | kernels, host | kernels, CUDA | `saro_oracle_model` |
    |---|---|---|---|
    | `cons!` | 0.183 ms | 0.163 ms | 0.151 ms |
    | `jac_coord!` | 3.06 ms | 0.246 ms | 1.68 ms |
    | `hess_coord!` | 33.2 ms | 0.641 ms | 41.3 ms |
    | `hprod!` | 5.21 ms | 0.292 ms | 42.1 ms |

    (SAROMM's figures are within 10% of these except `hprod!`, where the host
    port is the same 42 ms and the device 0.227 ms.)

    `cons!` gains nothing: 365 threads is six warps on a device with 80
    multiprocessors, and the work per thread is one stage evaluation. The
    second-order callbacks are where the thread count rises to 24090 and 4015,
    and `hprod!` compares against a host port that has no `hvp!` and so
    accumulates its 24090 coordinates against `v` on the host.

    On the host backend the narrow sweeps cost what they should: `hess_coord!`
    runs 66 stage evaluations where `saro_oracle_model` runs one
    `ForwardDiff.hessian`, and is still slightly faster, because 11-over-11
    nesting is 144 scalars per intermediate against four.

"""
function saro_kernel_model(name::AbstractString; T = Float64, backend = nothing,
                           prod::Bool = true)
    D = _saro_kernel_setup(name, T)
    d = _saro_kernel_data(D, T, backend)
    ka = backend === nothing ? KernelAbstractions.CPU() : backend
    nvar, ncon, nst = D.nvar, D.ncon, D.nst
    nlin = D.nlin

    # The declared Jacobian's linear block is constant in `x`; it lives on the
    # backend so filling it is a device-to-device copy rather than a host loop.
    lvald = ExaModels.convert_array(D.lval, backend)
    scratch = ExaModels.convert_array(zeros(T, 11 * nst), backend)

    # Each callback declares one-based indexing before it gathers: the decode's
    # indices are absolute positions in `1:nvar`, so an offset array would be
    # read at the wrong variables silently.
    f! = (res, xv) -> begin
        Base.require_one_based_indexing(res, xv)
        _saro_lin_kernel!(ka)(res, xv, d.rconst, d.lptr, d.lcol, d.lval;
                              ndrange = ncon)
        KernelAbstractions.synchronize(ka)
        _saro_cons_kernel!(ka)(res, xv, d.da, d.db, d.idx, d.rows; ndrange = nst)
        KernelAbstractions.synchronize(ka)
        return nothing
    end
    jac! = (vals, xv) -> begin
        Base.require_one_based_indexing(vals, xv)
        nlin == 0 || copyto!(view(vals, 1:nlin), lvald)
        _saro_jac_kernel!(ka)(vals, xv, d.da, d.db, d.idx, d.jpos;
                              ndrange = (11, nst))
        KernelAbstractions.synchronize(ka)
        return nothing
    end
    hess! = (vals, xv, yv) -> begin
        Base.require_one_based_indexing(vals, xv, yv)
        _saro_hess_kernel!(ka)(vals, xv, d.da, d.db, d.idx, d.rows, yv, d.pa,
                              d.pb; ndrange = (66, nst))
        KernelAbstractions.synchronize(ka)
        return nothing
    end
    hvp! = (Hv, xv, yv, vv) -> begin
        Base.require_one_based_indexing(Hv, xv, yv, vv)
        _saro_hvpblock_kernel!(ka)(scratch, xv, vv, d.da, d.db, d.idx, d.rows,
                                   yv; ndrange = (11, nst))
        KernelAbstractions.synchronize(ka)
        _saro_gather_kernel!(ka)(Hv, scratch, d.vptr, d.vsrc; ndrange = nvar)
        KernelAbstractions.synchronize(ka)
        return nothing
    end

    c = ExaModels.ExaCore(T; backend = backend)
    c, x = ExaModels.add_var(c, nvar; start = D.P.x0, lvar = D.P.lvar,
                             uvar = D.P.uvar)
    if !isempty(D.objterms)
        data = [(j = j, a = T(a)) for (j, a) in D.objterms]
        c, _ = ExaModels.add_obj(c, Base.Generator(t -> t.a * x[t.j], data))
    end
    vor = ExaModels.VectorNonlinearOracle(;
        nvar = nvar, ncon = ncon,
        jac_rows = D.jr, jac_cols = D.jc, hess_rows = D.hr, hess_cols = D.hc,
        lcon = D.lcon, ucon = D.ucon,
        f! = f!, jac! = jac!, hess! = hess!, hvp! = hvp!,
        adapt = Val(false))
    c = ExaModels.constraint(c, vor)
    return ExaModels.ExaModel(c; prod = prod)
end

"""
    saro_kernel_model(; T = Float64, backend = nothing, prod = true) -> ExaModel

SARO with its constraint callbacks as KernelAbstractions kernels. The
kernel-backed counterpart of [`saro_model`](@ref); see
[`saro_kernel_model(name)`](@ref).
"""
saro_kernel_model(; kwargs...) = saro_kernel_model("SARO"; kwargs...)

"""
    saromm_kernel_model(; T = Float64, backend = nothing, prod = true) -> ExaModel

SAROMM with its constraint callbacks as KernelAbstractions kernels. The
kernel-backed counterpart of [`saromm_model`](@ref); its 1095 extra `FPSM` /
`GPSM` rows are purely linear and are filled by the same row kernel as every
other row's linear part.
"""
saromm_kernel_model(; kwargs...) = saro_kernel_model("SAROMM"; kwargs...)
