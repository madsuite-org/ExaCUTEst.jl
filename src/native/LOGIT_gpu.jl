# AKIVA / HIELOW / PARKCH / STRATEC as ExaModels whose objective callbacks are
# KernelAbstractions kernels over the HieLoW log-likelihood, with the
# estimation sample resident on the model's backend.
#
# `logit_oracle_model` (LOGIT.jl) and `parkch_stratec_model`
# (PARKCH_STRATEC.jl) interface the same mathematics through host loops over
# `hielow_loglik`. The kernels here let these four models be built and
# evaluated with scalar indexing of device arrays disabled
# (`CUDA.allowscalar(false)`), which a host loop over a device array cannot be.
#
# ONE LOG-LIKELIHOOD, FOUR MODELS. All four SIF files append the same
# `SUBROUTINE HIELOW` -- Bierlaire's hierarchical (nested) logit -- and differ
# only in the tree and the sample. So `logit_kernel_model` serves all four and
# the thin wrappers below select the problem name, exactly as the two host
# builders share `hielow_loglik`.
#
# WHAT IS PARALLEL: RECORDS, NOT VARIABLES. These are maximum-likelihood
# objectives, a reduction over an estimation sample: 2 variables over 21
# records (AKIVA), 3 over 997 (HIELOW), 15 over 1386 (PARKCH), 10 over 1573
# (STRATEC). One thread per (output slot, record) pair, so the gradient runs
# `ncoef x nobs` threads and the Hessian `ncoef(ncoef+1)/2 x nobs`; each thread
# makes one sweep of one record's tree carrying at most four `ForwardDiff`
# components, and writes ONE partial result.
#
# THE REDUCTION IS TWO-STAGE AND USES NO ATOMICS. Stage one writes a
# `nslot x nobs` partial array -- one disjoint slot per thread, a pure write.
# Stage two runs one thread per slot and sums that slot's row sequentially in
# increasing record order. Both stages are therefore reproducible bit for bit
# run to run, and the summation order does not depend on the launch
# configuration or on the backend; the device and KA's CPU backend differ only
# where `exp`/`log1p` differ. One atomic per record per slot would instead
# serialise PARKCH's Hessian into 166320 read-modify-writes over 120 slots and
# its gradient into 20790 over 15, and make both scheduler-dependent. The
# absence of atomics also means these builders need no `isbitstype` gate:
# `KernelAbstractions.@atomic` has no `BigFloat` implementation, but nothing
# here asks for one, so `T = BigFloat` runs through the same kernels on the CPU
# backend.
#
# Stage two's cost is one thread per slot over `nobs` additions, i.e. a single
# thread for the objective. At these sample sizes that is microseconds and buys
# an order of summation that matches the host port's; a device `sum` over the
# partials would be faster asymptotically and reproducible only for a fixed
# array size.
#
# THE TREE IS A COMPILE-TIME CONSTANT. `_logit_tree` encodes one problem's tree
# -- each node's `THETA`, its parent, its utility terms as (coefficient,
# attribute) pairs and its children -- as a nested tuple that reaches the
# kernels as a `Val` type parameter, and `_logit_record_expr` emits the
# record's value loop unrolled against it. Every node, term and child index is
# then a literal and a record's utilities and inclusive values are plain locals
# in registers; the alternative is a per-thread scratch vector of length
# NNOEUD indexed at a runtime node number, which on a device means local
# memory. The cost is one kernel specialisation per problem.
#
# DERIVATIVES. `ForwardDiff` inside the kernel, over the same record function
# the value kernel uses. One thread computes ONE directional derivative:
# `grad` seeds one unit vector (two components per intermediate value), and
# `hess`/`hvp` nest two single-partial levels (four components) to get one
# mixed second derivative `sum_{k,l} d2f/db_k db_l p_k q_l`. Seeding a full
# `ncoef x ncoef` Hessian per record instead would carry 256 components per
# intermediate value for PARKCH and spill; computing the 120 lower-triangle
# entries as 120 four-component sweeps costs the same arithmetic as 15
# gradient-of-directional sweeps and keeps every thread's state in registers.
# The Fortran's hand-coded `GR`/`HESS` recursions are not ported, here or in
# the host ports.
#
# A SECOND TRANSCRIPTION OF THE VALUE LOOP, DELIBERATELY. `hielow_loglik`
# accumulates into two length-`NNOEUD` scratch vectors indexed by a runtime
# node number, which is what a kernel cannot do; `_logit_record` below is the
# same loop over tuples. The two are independent transcriptions of one Fortran
# routine, so they are compared against each other over every callback at two
# points in `test/logit_gpu_verify.jl` -- the host ports are the side that is
# checked against the compiled Fortran, so that comparison is what reaches the
# reference. `_logit_record` sums one record's conditional log-probabilities in
# the same order as the host (the path from the chosen alternative to the root
# visits nodes in increasing index order, because `ARBRE` guarantees parent >
# child), so per-record contributions agree to the bit and only the order of
# the outer sum over records differs.
#
# No `@inbounds`. The gathers and the sample reads keep their bounds checks;
# what `_logit_check_tables` establishes at build time is that they cannot
# fire, with a message naming the node or coefficient that is out of range
# instead of a kernel abort on the device.

using KernelAbstractions

# ---------------------------------------------------------------------------
# The tree as a compile-time constant.

"""
    _logit_tree(d::HieLoWData) -> Tuple

One HieLoW tree as a nested tuple, in node order: entry `nd` is
`(nd, theta, parent, terms, kids)` where `terms` holds node `nd`'s utility as
`(coefficient, attribute)` index pairs in the Fortran's `UTCOEF`/`UTCARA`
order and `kids` its children in increasing order.

The node index rides along in the entry so that the helpers below can recurse
by CONSUMING the tuple and never by incrementing a counter in the type domain:
a recursion whose type parameter grows -- `Val(ND + 1)` -- makes inference give
up and widen to `Any`, which on a device is an `InvalidIRError` rather than a
slowdown.

Passed to the kernels inside a `Val`, so every index in it is a literal in the
generated code and the per-record sweep needs no mutable scratch. `theta` and
`parent` select the branches of [`_logit_record`](@ref) at compile time:
`theta == -1` marks a leaf, `-2` the root and `k > 0` a structural node scaled
by `beta[k]`; `parent == -1` marks the root.
"""
_logit_tree(d::HieLoWData) = Tuple(
    (nd, d.theta[nd], d.arbre[nd],
     Tuple((d.utcoef[nd, i], d.utcara[nd, i]) for i in 1:d.utleng[nd]),
     Tuple(d.children[nd]))
    for nd in 1:d.nnoeud)

"""
    _logit_pathmasks(d::HieLoWData) -> Vector{Int32}

Bit `nd - 1` of entry `obs` is set when node `nd` contributes a conditional
log-probability to record `obs`, i.e. when `nd` is the chosen alternative or
one of its ancestors below the root.

The walk from the chosen alternative to the root is what the Fortran's value
loop ends with. Precomputing it as a mask lets the kernel visit nodes in
increasing index order -- the order the walk itself takes, since `ARBRE`
guarantees parent > child -- without indexing a tuple at a runtime node
number.
"""
function _logit_pathmasks(d::HieLoWData)
    d.nnoeud <= 31 || error("the path mask holds at most 31 nodes, not $(d.nnoeud)")
    pm = zeros(Int32, d.nobs)
    for obs in eachindex(pm)
        nd = d.chosen[obs]
        m = Int32(0)
        while d.arbre[nd] != -1
            m |= Int32(1) << (nd - 1)
            nd = d.arbre[nd]
        end
        pm[obs] = m
    end
    return pm
end

# ---------------------------------------------------------------------------
# The log-likelihood of ONE record.

"""
    _logit_record_expr(tree) -> Expr

`SUBROUTINE HIELOW`'s value loop for one observation, emitted as straight-line
code for the given tree: one local per node for its utility and one for its
inclusive value, with every node, coefficient and attribute index a literal.

The loop structure, the order of the additions and the strict comparison that
picks the largest child utility are [`hielow_loglik`](@ref)'s, with the loops
unwound; the generated code is what makes the per-record utilities and
inclusive values live in registers, since a device has no cheap per-thread
scratch vector to index at a runtime node number.

Emitting the body is also the only way to keep it type-stable. Walking the
tree by recursion over a `Val` type parameter -- even one that only shrinks --
loses inference at the THIRD level: `Base.return_types` widens to `Any`, which
costs the host a dynamic dispatch and the device an `InvalidIRError`.
"""
function _logit_record_expr(tree)
    u(nd) = Symbol(:u_, nd)
    l(nd) = Symbol(:l_, nd)
    body = Expr(:block, Expr(:meta, :inline), :(z = zero(eltype(beta))))
    for (nd, theta, parent, terms, kids) in tree
        # Elemental utility: linear in the coefficients, the record's
        # attributes as data. The root has none -- the Fortran skips the block
        # when ARBRE(NOEUD) == -1 -- and a structural node's own terms are
        # empty. `oftype` widens an attribute to the differentiation type
        # exactly as the host port's `T(d.caract[obs, k])` does.
        acc = :z
        if parent != -1
            for (c, k) in terms
                acc = :($acc + beta[$c] * oftype(z, caract[obs, $k]))
            end
        end
        push!(body.args, :($(u(nd)) = $acc))
        theta == -1 && continue

        # Inclusive value over the available children, shifted by the largest
        # of them so that nothing overflows. `kmax` ends positive because every
        # non-leaf has an available child at every record, which
        # `hielow_parse` establishes; with none the inclusive value would be
        # log(0).
        km, mx, sm = Symbol(:kmax_, nd), Symbol(:m_, nd), Symbol(:s_, nd)
        push!(body.args, :($km = 0), :($mx = z))
        for c in kids
            push!(body.args, quote
                      if disp[obs, $c] && ($km == 0 || $(u(c)) > $mx)
                          $km = $c
                          $mx = $(u(c))
                      end
                  end)
        end
        acc = :z
        for c in kids
            acc = :($acc + (disp[obs, $c] && $c != $km ? exp($(u(c)) - $mx) : z))
        end
        push!(body.args, :($sm = $acc), :($(l(nd)) = $mx + log1p($sm)))
        if theta > 0
            push!(body.args, :($(u(nd)) = $(u(nd)) + beta[$theta] * $(l(nd))))
        end
    end

    # One conditional log-probability per level of the chosen alternative's
    # path to the root. `ARBRE` guarantees parent > child, so the walk visits
    # nodes in increasing index order and this loop accumulates them in the
    # same order the host port does.
    push!(body.args, :(like = z))
    for (nd, _, parent, _, _) in tree
        parent == -1 && continue
        push!(body.args, quote
                  if (pm >> $(nd - 1)) & Int32(1) != 0
                      like = like + ($(u(nd)) - $(l(parent)))
                  end
              end)
    end
    push!(body.args, :(return like))
    return body
end

"""
    _logit_record(Val(tree), beta, caract, disp, pm, obs) -> eltype(beta)

Record `obs`'s contribution to the HieLoW log-likelihood at coefficients
`beta`: the sum of one conditional-choice log probability per level of the
chosen alternative's path to the root.

Generic in `eltype(beta)`, so the same code gives the value in `Float64`,
`Float32` or `BigFloat` and the derivatives through `ForwardDiff.Dual`. `pm`
is the record's path mask ([`_logit_pathmasks`](@ref)); the body is
[`_logit_record_expr`](@ref) for `tree`.

Inclusive values are formed by shifting by the maximum child utility, as in
[`hielow_loglik`](@ref) and for the same reason: nothing overflows, so the
Fortran's `LOGGRA`/`GRAND` clipping is not needed.
"""
@generated function _logit_record(::Val{TREE}, beta::NTuple{NC, D}, caract, disp,
                                  pm, obs) where {TREE, NC, D}
    return _logit_record_expr(TREE)
end

# ---------------------------------------------------------------------------
# Directional derivatives of one record, by hand-written ForwardDiff seeding.
# `ForwardDiff.gradient`/`hessian` take an `AbstractVector`, allocate a result
# and consult a mutable tag counter, none of which a kernel has.

struct _LogitTagInner end
struct _LogitTagOuter end

"The unit vector `e_a`, shaped and typed like the coefficient tuple `b`."
@inline _logit_unit(b::NTuple{NC, T}, a) where {NC, T} =
    ntuple(k -> ifelse(k == a, one(T), zero(T)), Val(NC))

"""
    _logit_d1(Val(tree), beta, p, caract, disp, pm, obs)

The directional derivative `sum_k p_k df/dbeta_k` of one record, by one level
of forward-mode AD over [`_logit_record`](@ref).
"""
@inline function _logit_d1(::Val{TREE}, beta::NTuple{NC, T}, p::NTuple{NC, T},
                           caract, disp, pm, obs) where {TREE, NC, T}
    D = ForwardDiff.Dual{_LogitTagInner, T, 1}
    bd = ntuple(k -> D(beta[k], ForwardDiff.Partials{1, T}((p[k],))), Val(NC))
    return ForwardDiff.partials(
        _logit_record(Val(TREE), bd, caract, disp, pm, obs))[1]
end

"""
    _logit_d2(Val(tree), beta, p, q, caract, disp, pm, obs)

The second mixed directional derivative
`sum_{k,l} p_k q_l d2f/(dbeta_k dbeta_l)` of one record, by nesting two
single-partial levels of forward-mode AD over [`_logit_record`](@ref).

With `p = e_a`, `q = e_b` this is one Hessian entry; with `p = e_a`, `q = v` it
is one component of `H * v`. Four components per intermediate value either
way.
"""
@inline function _logit_d2(::Val{TREE}, beta::NTuple{NC, T}, p::NTuple{NC, T},
                           q::NTuple{NC, T}, caract, disp, pm, obs) where {TREE, NC, T}
    DI = ForwardDiff.Dual{_LogitTagInner, T, 1}
    DO = ForwardDiff.Dual{_LogitTagOuter, DI, 1}
    z = ForwardDiff.Partials{1, T}((zero(T),))
    bd = ntuple(k -> DO(DI(beta[k], ForwardDiff.Partials{1, T}((p[k],))),
                        ForwardDiff.Partials{1, DI}((DI(q[k], z),))), Val(NC))
    r = _logit_record(Val(TREE), bd, caract, disp, pm, obs)
    return ForwardDiff.partials(ForwardDiff.partials(r)[1])[1]
end

@inline _logit_gather(x, idx::NTuple{NC, <:Integer}) where {NC} =
    ntuple(a -> x[idx[a]], Val(NC))

# ---------------------------------------------------------------------------
# Kernels. One thread per (output slot, record); `part[slot, record]` is that
# thread's own entry, so stage one never accumulates and never needs an atomic.
#
# `idx` is passed by value as a tuple: `ncoef <= 15`, so the gather indices
# live in registers. `tree` is a `Val` and carries no data at run time.
#
# Every index these kernels form is in range, which `_logit_check_tables`
# establishes once at build time: `idx` into `1:nvar`, the tree's coefficient
# indices into `1:ncoef`, its attribute indices into `1:ncarac`, and the
# sample's own extents.

@kernel function _logit_val_kernel!(part, @Const(x), idx, tree, @Const(caract),
                                    @Const(disp), @Const(pmask))
    i = @index(Global)
    part[1, i] = _logit_record(tree, _logit_gather(x, idx), caract, disp,
                               pmask[i], i)
end

@kernel function _logit_grad_kernel!(part, @Const(x), idx, tree, @Const(caract),
                                     @Const(disp), @Const(pmask))
    a, i = @index(Global, NTuple)
    b = _logit_gather(x, idx)
    part[a, i] = _logit_d1(tree, b, _logit_unit(b, a), caract, disp, pmask[i], i)
end

# Slot `p` of the dense lower triangle holds entry `(ltr[p], ltc[p])` of the
# coefficient block, in the order `logit_kernel_model` declares.
@kernel function _logit_hess_kernel!(part, @Const(x), idx, tree, @Const(caract),
                                     @Const(disp), @Const(pmask), @Const(ltr),
                                     @Const(ltc))
    p, i = @index(Global, NTuple)
    b = _logit_gather(x, idx)
    part[p, i] = _logit_d2(tree, b, _logit_unit(b, ltc[p]), _logit_unit(b, ltr[p]),
                           caract, disp, pmask[i], i)
end

@kernel function _logit_hvp_kernel!(part, @Const(x), @Const(vv), idx, tree,
                                    @Const(caract), @Const(disp), @Const(pmask))
    a, i = @index(Global, NTuple)
    b = _logit_gather(x, idx)
    part[a, i] = _logit_d2(tree, b, _logit_unit(b, a), _logit_gather(vv, idx),
                           caract, disp, pmask[i], i)
end

# Stage two: one thread per output slot, summing that slot's partials in
# increasing record order. Sequential rather than a tree so that the order is
# the same on every backend and on every launch, and `w` -- the objective
# group's weight on the element -- scales each slot once instead of every
# record's contribution.
#
# `out` is ASSIGNED, not accumulated: `dest` is injective, so each slot of
# `out` that any thread names is named by exactly one thread. Slots of `out`
# that `dest` does not name keep what the caller put there.
@kernel function _logit_reduce_kernel!(out, @Const(part), @Const(dest), w)
    p = @index(Global)
    s = zero(eltype(part))
    for i in axes(part, 2)
        s += part[p, i]
    end
    out[dest[p]] = w * s
end

# ---------------------------------------------------------------------------
# Device-resident sample.

"""
    HieLoWKernelData

One HieLoW problem's estimation sample as a struct of arrays on whatever
device the model is built for: `caract` is nobs x ncarac attributes, `disp` the
nobs x nnoeud availability mask and `pmask` one path mask per record
([`_logit_pathmasks`](@ref)). `ltr`/`ltc` are the row and column of each dense
lower-triangle Hessian slot, `hslot` the identity map those slots reduce
through and `dest` the variable index each coefficient scatters to.

The array fields carry their own type parameters, so the same struct holds host
arrays or a device backend's arrays and the kernels take their arguments out of
it unchanged.

The tree itself is NOT here: it is a `Val` type parameter
([`_logit_tree`](@ref)), so it reaches the kernels as part of their signature.
"""
struct HieLoWKernelData{MT, MB, VI}
    caract::MT
    disp::MB
    pmask::VI
    ltr::VI
    ltc::VI
    hslot::VI
    dest::VI
    nobs::Int
    ncoef::Int
    ntri::Int
end

"""
    _logit_check_tables(name, d, idx, nvar)

Check at build time that every index the kernels form is in range: the
elemental variables `idx` into `1:nvar` and distinct, each node's utility
coefficients into `1:ncoef`, its attributes into `1:ncarac`, its children and
parent into `1:nnoeud`, and each record's chosen alternative likewise.

The kernels keep their bounds checks, so this does not make anything legal; it
makes the diagnosis useful. An index that the parse got wrong is reported here
naming the node and the coefficient, where inside a kernel it surfaces as an
exception from a device launch with nothing pointing back at the sample.
`hielow_parse` has already checked the tree's own invariants -- one root,
parent > child, every non-leaf with an available child at every record -- which
[`_logit_record`](@ref) relies on.
"""
function _logit_check_tables(name::AbstractString, d::HieLoWData,
                             idx::NTuple{NC, <:Integer}, nvar::Integer) where {NC}
    NC == d.ncoef ||
        error("$name: $NC elemental variables for $(d.ncoef) coefficients")
    seen = falses(nvar)
    for (a, j) in pairs(idx)
        1 <= j <= nvar ||
            error("$name: coefficient $a maps to variable $j, outside 1:$nvar")
        seen[j] && error("$name: coefficient $a repeats variable index $j")
        seen[j] = true
    end
    ncarac = size(d.caract, 2)
    for nd in 1:d.nnoeud
        for i in 1:d.utleng[nd]
            c, k = d.utcoef[nd, i], d.utcara[nd, i]
            1 <= c <= d.ncoef || error(
                "$name: node $nd term $i uses coefficient $c, outside 1:$(d.ncoef)")
            1 <= k <= ncarac || error(
                "$name: node $nd term $i uses attribute $k, outside 1:$ncarac")
        end
        d.theta[nd] <= d.ncoef || error(
            "$name: THETA($nd) = $(d.theta[nd]) exceeds $(d.ncoef) coefficients")
        for c in d.children[nd]
            1 <= c < nd || error("$name: node $nd has child $c")
        end
        d.arbre[nd] == -1 || 1 <= d.arbre[nd] <= d.nnoeud ||
            error("$name: ARBRE($nd) = $(d.arbre[nd]) outside 1:$(d.nnoeud)")
    end
    size(d.disp) == (d.nobs, d.nnoeud) ||
        error("$name: DISP is $(size(d.disp)), not $((d.nobs, d.nnoeud))")
    for obs in 1:d.nobs
        1 <= d.chosen[obs] <= d.nnoeud || error(
            "$name: CHOSEN($obs) = $(d.chosen[obs]) outside 1:$(d.nnoeud)")
    end
    return nothing
end

function _logit_kernel_data(d::HieLoWData, idx, ::Type{T}, backend) where {T}
    nc = d.ncoef
    cv = v -> ExaModels.convert_array(v, backend)
    ltr = Int32[a for a in 1:nc for _ in 1:a]
    ltc = Int32[b for a in 1:nc for b in 1:a]
    return HieLoWKernelData(
        cv(T[T(v) for v in d.caract]), cv(Matrix{Bool}(d.disp)),
        cv(_logit_pathmasks(d)), cv(ltr), cv(ltc),
        cv(Int32[p for p in eachindex(ltr)]), cv(Int32[idx[a] for a in 1:nc]),
        d.nobs, nc, length(ltr))
end

# ---------------------------------------------------------------------------

"""
    logit_kernel_model(name; T = Float64, backend = nothing) -> ExaModel

Build AKIVA, HIELOW, PARKCH or STRATEC as an `ExaModel` whose objective
callbacks are KernelAbstractions kernels over the HieLoW log-likelihood, with
the estimation sample resident on `backend`.

`backend = nothing` builds a host model that runs the same kernels on
KernelAbstractions' CPU backend; `backend = CUDABackend()` builds a device
model that never indexes a device array from the host, so it is usable with
`CUDA.allowscalar(false)`.

Structure (variables, bounds, start point, the single objective group and its
element's variable map) comes from `sif_decode(...; external = :defer)`; the
tree and the estimation sample come from [`hielow_parse`](@ref) reading the
`DATA` statements in the same file, converted once here and passed to every
launch as kernel arguments. No `sifdecoder` and no Fortran toolchain.

All four problems are unconstrained with a dense `ncoef x ncoef` coefficient
block, so the objective goes in as a single `ScalarNonlinearOracle` with `f`,
`grad!`, `hvp!` and an explicit `hess!` whose declared sparsity is the dense
lower triangle -- 3, 6, 120 and 55 entries, with no repeated coordinate.
Supplying `hess!` is what keeps `hess_coord` on the device: without it
ExaModels rebuilds the pattern from one `hvp!` per column, indexing the value
array slot by slot.

`T` may be any of `Float64`, `Float32` or `BigFloat`: the reduction uses no
`KernelAbstractions.@atomic`, so there is no bits-type restriction to route
around. A `BigFloat` model necessarily runs on the CPU backend.

The host builders [`logit_oracle_model`](@ref) and
[`parkch_stratec_model`](@ref) evaluate the same mathematics through
`hielow_loglik`, and are the side checked against the compiled Fortran;
`test/logit_gpu_verify.jl` compares these kernels against them.
"""
function logit_kernel_model(name::AbstractString; T = Float64, backend = nothing)
    P = ExaModelsSIF.sif_decode(sif_problem_path(name); external = :defer)
    haskey(P.external_types, "LIKE") ||
        error("$name: expected a deferred LIKE element type, found " *
              "$(collect(keys(P.external_types)))")
    length(P.grouporder) == 1 ||
        error("$name: expected 1 group, found $(length(P.grouporder))")
    g = P.groups[only(P.grouporder)]
    (g.kind == 'N' && isempty(g.lin) && length(g.elems) == 1 && g.scale == 1 &&
     g.range_ === nothing) || error("$name: unexpected objective group shape")
    en, w = only(g.elems)
    e = P.elements[en]
    d = hielow_data(name)

    # BETA(i) of the Fortran is the element's i-th elemental variable B<i>.
    idx = ntuple(i -> e.varmap[Symbol(:B, i)], d.ncoef)
    n = length(P.varnames)
    _logit_check_tables(name, d, idx, n)

    tree = Val(_logit_tree(d))
    dat = _logit_kernel_data(d, idx, T, backend)
    ka = backend === nothing ? KernelAbstractions.CPU() : backend
    nc, ntri, nobs = dat.ncoef, dat.ntri, dat.nobs
    wT, cT = T(w), T(g.constant)

    # Per-(slot, record) partials, converted once. Stage one fills these and
    # stage two reduces them; separate buffers per callback, so a `grad!`
    # cannot land in the middle of an `hvp!`'s partials.
    cv = v -> ExaModels.convert_array(v, backend)
    pval = cv(zeros(T, 1, nobs))
    pgrad = cv(zeros(T, nc, nobs))
    phess = cv(zeros(T, ntri, nobs))
    phvp = cv(zeros(T, nc, nobs))
    one1 = cv(Int32[1])
    fout = cv(zeros(T, 1))

    f = xv -> begin
        Base.require_one_based_indexing(xv)
        _logit_val_kernel!(ka)(pval, xv, idx, tree, dat.caract, dat.disp,
                               dat.pmask; ndrange = nobs)
        _logit_reduce_kernel!(ka)(fout, pval, one1, wT; ndrange = 1)
        KernelAbstractions.synchronize(ka)
        return only(Array(fout)) - cT
    end
    grad! = (gv, xv) -> begin
        Base.require_one_based_indexing(gv, xv)
        fill!(gv, zero(eltype(gv)))
        _logit_grad_kernel!(ka)(pgrad, xv, idx, tree, dat.caract, dat.disp,
                                dat.pmask; ndrange = (nc, nobs))
        _logit_reduce_kernel!(ka)(gv, pgrad, dat.dest, wT; ndrange = nc)
        KernelAbstractions.synchronize(ka)
        return gv
    end
    hess! = (vals, xv) -> begin
        Base.require_one_based_indexing(vals, xv)
        _logit_hess_kernel!(ka)(phess, xv, idx, tree, dat.caract, dat.disp,
                                dat.pmask, dat.ltr, dat.ltc;
                                ndrange = (ntri, nobs))
        _logit_reduce_kernel!(ka)(vals, phess, dat.hslot, wT; ndrange = ntri)
        KernelAbstractions.synchronize(ka)
        return vals
    end
    hvp! = (Hv, xv, vv) -> begin
        Base.require_one_based_indexing(Hv, xv, vv)
        fill!(Hv, zero(eltype(Hv)))
        _logit_hvp_kernel!(ka)(phvp, xv, vv, idx, tree, dat.caract, dat.disp,
                              dat.pmask; ndrange = (nc, nobs))
        _logit_reduce_kernel!(ka)(Hv, phvp, dat.dest, wT; ndrange = nc)
        KernelAbstractions.synchronize(ka)
        return Hv
    end

    # Dense lower triangle of the coefficient block, in the slot order
    # `_logit_hess_kernel!` writes. Every coefficient enters every record's
    # utility -- through the nest scale if nowhere else -- so the block is
    # dense, and no (row, col) pair repeats.
    hr = [max(idx[a], idx[b]) for a in 1:nc for b in 1:a]
    hc = [min(idx[a], idx[b]) for a in 1:nc for b in 1:a]

    c = ExaModels.ExaCore(T; backend = backend)
    c, _ = ExaModels.add_var(c, n; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    c = ExaModels.objective(c, ExaModels.ScalarNonlinearOracle(;
        nvar = n, f = f, grad! = grad!, hvp! = hvp!, hess! = hess!,
        hess_rows = hr, hess_cols = hc, adapt = Val(false)))
    return ExaModels.ExaModel(c)
end

"""
    akiva_kernel_model(; T = Float64, backend = nothing) -> ExaModel

AKIVA -- Ben-Akiva and Lerman's 21-record binary logit in 2 coefficients -- as
an `ExaModel` with kernel objective callbacks. See
[`logit_kernel_model`](@ref).
"""
akiva_kernel_model(; kwargs...) = logit_kernel_model("AKIVA"; kwargs...)

"""
    hielow_kernel_model(; T = Float64, backend = nothing) -> ExaModel

HIELOW -- a 997-record two-level nested logit in 3 coefficients -- as an
`ExaModel` with kernel objective callbacks. See [`logit_kernel_model`](@ref).
"""
hielow_kernel_model(; kwargs...) = logit_kernel_model("HIELOW"; kwargs...)

"""
    parkch_kernel_model(; T = Float64, backend = nothing) -> ExaModel

PARKCH -- a 1386-record three-level nested logit in 15 coefficients -- as an
`ExaModel` with kernel objective callbacks. See [`logit_kernel_model`](@ref).
"""
parkch_kernel_model(; kwargs...) = logit_kernel_model("PARKCH"; kwargs...)

"""
    stratec_kernel_model(; T = Float64, backend = nothing) -> ExaModel

STRATEC -- a 1573-record nested logit over three nests sharing one log-sum
scale, in 10 coefficients -- as an `ExaModel` with kernel objective callbacks.
See [`logit_kernel_model`](@ref).
"""
stratec_kernel_model(; kwargs...) = logit_kernel_model("STRATEC"; kwargs...)
