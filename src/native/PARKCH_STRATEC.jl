# PARKCH / STRATEC: the two largest SIF files in the collection, as native
# ExaModels.
#
# WHAT THE FORTRAN COMPUTES. Both files append `SUBROUTINE HIELOW(CGRAD,CHESS)`
# behind three wrappers `F`, `G(.,I)`, `H(.,I,J)` that pass the elemental
# variables `B1..B<NCOEF>` through `COMMON /COEF/`. It is Bierlaire's
# hierarchical (nested) logit log-likelihood -- the SAME routine AKIVA and
# HIELOW carry, already ported in `src/native/LOGIT.jl`. That is not a guess:
# strip comments, `DATA` lines and the `DISP(i,j) = .FALSE.` overrides from the
# subroutine in all three files and the remaining 204 lines are byte-identical
# except for
#
#   * the `PARAMETER` sizes, and
#   * ONE line, in the HESSIAN branch only: HIELOW has
#     `ELSE IF (THETA(NOEUD) .NE. -1)` where PARKCH has
#     `ELSE IF (THETA(NOEUD) .GT. 0)` (STRATEC agrees with HIELOW).
#
# The differing line is inside `IF (CHESS)`, i.e. in the hand-coded second
# derivative recursion, which this port does not use at all. The VALUE loop --
# lines 55..108 of the stripped source -- is identical in all three. (AKIVA is
# the odd one out of the family: it passes `BETA/LIKE/GR/HESS` as arguments
# instead of through `COMMON`, so its subroutine was not part of this diff.)
#
# So the mathematics is `hielow_loglik` in `src/native/LOGIT.jl`, and this file
# deliberately REUSES it rather than re-translating it: it is the same Fortran,
# it is already checked against two problems and two hand-derived closed forms,
# and a second transcription of the same loop would only add a second place to
# be wrong. What is new here is the two problems' structure, the precision
# audit, a faster second-derivative path, and the verification.
#
#   PARKCH   15 coefficients (BET1..BET14, THE1), 1386 observations,
#            20 attributes, a 7-node THREE-level tree: leaves 1 (public
#            transport), 2 (outside the ring), 3 (illegal) and nest 6 under
#            root 7; leaves 4 (off-street) and 5 (on-street) under nest 6,
#            whose log-sum scale is THE1 = BETA(15). 361 of the 1386
#            respondents cannot choose node 2 (`DISP(obs,2) = .FALSE.`).
#            Stated-preference parking survey, Namur.
#   STRATEC  10 coefficients (BET1..BET9, THE1), 1573 observations, 11
#            attributes, a 10-node tree: leaves {1,2} under nest 7, leaf {3}
#            under nest 8, leaves {4,5,6} under nest 9, the three nests under
#            root 10, ALL THREE scaled by the same THE1 = BETA(10). Every
#            alternative is available to everyone. Parking choice, Brussels.
#            Node 6 is a leaf with `UTLENG = 0`, i.e. the reference
#            alternative with utility identically zero, and node 8 is a nest
#            with a SINGLE child -- see "where this port differs" below.
#
# SHAPE: GENERIC JULIA FUNCTION + `ScalarNonlinearOracle`, same as AKIVA and
# HIELOW. Folding is impossible (the element value depends on the variables),
# and as an ExaModels expression the value would be a sum over 1386 (resp.
# 1573) observations, each of ~40 operations -- tens of thousands of nested
# type parameters, where ~30 already failed to compile in an hour (src/BA.jl).
# Derivatives are ForwardDiff over the value function. The SIF's hand-coded
# `G` (Daly's recursions) and `H` routines are NOT ported; they are the
# reference, not the source.
#
# PRECISION: THE DEFAULT-REAL TRAP IS ABSENT HERE, AND THE DATA PROVES IT
# RATHER THAN THE SOURCE MERELY SUGGESTING IT. `SUBROUTINE HIELOW` declares
# `IMPLICIT NONE` and every variable, so no scalar is silently single
# precision. That leaves the constants, which is what cost HIELOW 8e-11:
# HIELOW writes all 5982 of its attributes as bare reals (`3.8712`), which
# Fortran loads as default REAL, i.e. Float32. PARKCH and STRATEC write all
# 27720 and 17303 of theirs as INTEGER constants -- `DATA CARACT(1,2) /100/`,
# never `/100./` and never `/1.0D2/`. An integer constant converts to double
# exactly, so there is nothing to round; and the two readings are not merely
# close but IDENTICAL, since the 27 (PARKCH) and 11 (STRATEC) distinct values
# are the integers 0..200 and 0..225, each exact in Float32
# (`all(x -> Float64(Float32(x)) == x, d.caract)` is `true` for both). The
# question that discriminated on HIELOW therefore cannot discriminate here,
# which is the honest statement: Float32-rounding the literals changes
# nothing, bit for bit, so this port does it and it makes no difference.
# `_hielow_float` in LOGIT.jl already distinguishes the integer case.
#
# WHY THE FORTRAN IS SO SLOW -- it is a one-line bug, not the arithmetic.
# `G` and `H` were written to cache: `COMMON /COEF/` keeps `COMPG`/`COMPH` and
# the point `BETA`, and the routines mean to return `GR(I)` / `HESS(I,J)`
# unchanged when asked twice at the same point. But both open with
#
#       COMPG = .TRUE.                 (resp. COMPH = .TRUE.)
#       IF (.NOT. COMPG) THEN  ... return the cached value ...
#
# -- the flag is set immediately before being tested for the opposite value,
# so the cache arm is dead code and every call recomputes. The SIF's
# `INDIVIDUALS` block lists `G B1 .. G B15` and all 120 lower-triangle `H Bi
# Bj` entries as separate external calls, so CUTEst pays a full
# 1386-observation sweep 15 times for one gradient and 120 times for one
# Hessian, each Hessian sweep itself an `I,J` double loop over the 15x15 pairs.
# That is the ~0.55 s `hprod`. The port computes the whole Hessian in one AD
# pass, so the speedup is partly better derivatives and partly not repeating
# the same sweep 120 times.
#
#   measured on shin-compute-001, Float64, medians of nine calls each at a
#   different point so neither cache can be hit (~/git/scratch/ps_timing.jl):
#
#              PARKCH (n = 15)              STRATEC (n = 10)
#              CUTEst    native    x        CUTEst    native    x
#   obj        0.21 ms   0.28 ms   0.8      0.25 ms   0.35 ms   0.7
#   grad       6.91 ms   0.60 ms  11.5      5.09 ms   0.49 ms  10.4
#   hess     505.25 ms  17.19 ms  29.4    168.68 ms   6.51 ms  25.9
#   hprod    487.53 ms  17.89 ms  27.3    167.91 ms   6.54 ms  25.7
#
# Read honestly, because the mechanism matters more than the ratio. The VALUE
# is not faster: the Fortran's value loop is tight and the port is 20-40%
# behind it. Nor is the port's AD arithmetic faster than the Fortran's
# hand-coded recursions -- divide the CUTEst column by the number of sweeps the
# dead cache forces and ONE Fortran sweep is 6.91/15 = 0.46 ms for a gradient
# and 505/120 = 4.2 ms for a Hessian, against the port's 0.60 ms and 17.2 ms.
# (Those per-sweep costs are 2.2x and 20x the 0.21 ms value sweep, which is
# about what the `CGRAD` branch and the 120-pair `I,J` double loop
# respectively add -- independent support for the 15-and-120 count.) So the
# ENTIRE speedup is "evaluate once instead of 15 or 120 times", and a
# three-line fix to the Fortran's `COMPG`/`COMPH` test would take most of it
# back. It is nonetheless the real cost CUTEst charges today, and 0.5 s per
# Hessian is what times these two problems out under a solver.
#
# SECOND DERIVATIVES: ONE FULL HESSIAN, CACHED, NOT n HESSIAN-VECTOR PRODUCTS.
# ExaModels reconstructs a `ScalarNonlinearOracle`'s sparse Hessian by calling
# `hvp!` once per distinct column (`_hess_reconstruct_via_hvp!`), so with
# `n = 15` a naive `hvp!` that forms the full Hessian each time would compute
# it 15 times over. Three options were measured on PARKCH:
#
#   ForwardDiff.hessian, default chunk (8)        17.9 ms
#   ForwardDiff.hessian, full chunk (15)          11.5 ms
#   forward-over-forward directional H*v           1.9 ms  (x15 = 29 ms)
#
# so the full-chunk Hessian wins for `hess_coord` and the directional product
# wins for a lone `hprod`. This port keeps the dense Hessian in the closure,
# keyed on the point, and answers `hvp!` with a matrix-vector product: one AD
# pass serves all 15 columns, `hess_coord` costs 17.2 ms end to end instead of
# the ~29 ms the directional product would cost, and a lone `hprod` costs
# 17.9 ms instead of 1.9 ms. That is the same trade the
# Fortran's `COMPG`/`COMPH` flags were written to make -- here it works. The
# value function itself stays pure; the cache lives in the model, so a model
# is not safe to evaluate from two threads at once (neither is the Fortran:
# its cache is a `COMMON` block).
#
# WHERE THIS PORT DIFFERS FROM THE FORTRAN, at the 1e-16 level and knowingly.
# `hielow_loglik` forms a nest's inclusive value by shifting by the maximum
# child utility, `m + log1p(sum_{c != argmax} exp(u_c - m))`, where the
# Fortran forms `LOG(sum_c EXP(u_c))` with the children summed in index order
# and clipped at `GRAND = 1.0D308`. Two consequences here:
#
#   * STRATEC's node 8 has ONE child, so the Fortran computes `LOG(EXP(u_3))`
#     -- two transcendentals that need not cancel -- where this port returns
#     `u_3 + log1p(0) = u_3` exactly. The port is the more accurate of the two
#     by construction. MEASURED, though, the effect is nil: running the
#     hand-derived closed form both ways at the perturbed point gives the same
#     Float64 to the last bit (relative difference 0.0), because `u_3 =
#     b7*CARACT(obs,8)` is small and `log(exp(.))` round-trips there. So this
#     is a difference in the arithmetic that the data does not expose, not a
#     measured discrepancy -- stated that way rather than claimed as the
#     source of the 1e-15 residual.
#   * the clipping differs in the overflow regime (`|u| >= 709.196`), exactly
#     as documented in LOGIT.jl. Unreachable at any sane iterate: STRATEC's
#     largest attribute is 225, so it needs `|beta| ~ 3`.
#
# The first is measured in the verification script rather than asserted. The
# second is NOT probed here: LOGIT.jl's own verification already exhibits it
# on HIELOW, the behaviour is a property of `hielow_loglik` and not of these
# two problems, and no iterate a solver would visit reaches it.
#
# MEASURED AGREEMENT (~/git/scratch/parkch_stratec_verify.jl, and its .tsv).
# Against a real `CUTEstModel`, permuted by SIF variable name, at x0 and at a
# perturbed point; worst relative error over the two points:
#
#                                    PARKCH     STRATEC
#   obj                              6.5e-15    1.2e-15
#   grad                             6.1e-16    4.8e-15
#   hprod, n coordinate + 3 random   1.2e-14    7.1e-14
#   hess, FULL n x n matrix          2.6e-15    6.4e-14
#   value vs hand-derived closed form 3.8e-15   7.6e-15
#   value vs BigFloat                9.3e-16    6.2e-15
#
# `cons`/`jac` are vacuous: both problems are unconstrained (`OUR2-RN-15-0`,
# `OUR2-RN-10-0`, `ncon = 0` on both sides). The classification's fourth
# character is `2` and the SIF's `INDIVIDUALS` block does supply `H`, so the
# reference Hessian is a real one and comparing against it is meaningful --
# unlike SARO. It was additionally checked against a central difference of
# CUTEst's own gradient (1e-7 .. 2e-8, the accuracy a central difference can
# deliver), which agrees with the direct comparison and so confirms that the
# two are not agreeing on a shared mistake in the `H` routine.
#
# The residual 1e-14 .. 1e-15 is the roundoff floor of a 1386-term (1573-term)
# sum evaluated in two different orders, not a disagreement: at the perturbed
# point PARKCH's objective is 5877.519962470737 (port) against
# 5877.519962470775 (Fortran) and 5877.519962470731 in BigFloat, i.e. the port
# is 5.5e-12 ABSOLUTE from the exact value and the Fortran is 4.4e-11, so the
# port is the closer of the two.
#
# A GPU BACKEND GOES THROUGH `logit_kernel_model` (src/native/LOGIT_gpu.jl),
# which evaluates the same log-likelihood in KernelAbstractions kernels over
# records and serves all four HieLoW problems. The callbacks here are host
# loops, and the Hessian cache below is a mutable host array, so they refuse a
# backend.

"""
    _hielow_gather!(b, x, idx) -> b

Copy the `ncoef` elemental variables out of the full variable vector: `b[a] =
x[idx[a]]`, converted to `b`'s element type.

A free function rather than a closure body on purpose. `T` reaches
[`parkch_stratec_model`](@ref) as a keyword VALUE, so inside its closures
`T(...)` is a call on a `DataType` variable and inference gives up; here `T`
is a real type parameter and the conversion is static.

Worth being exact about the payoff, which is smaller than the reasoning
suggests: moving these two loops out of the closures took STRATEC's gradient
from 1.03 ms to 0.58 ms and PARKCH's `hess_coord` from 16.7 ms to 15.2 ms,
and the second of those is inside run-to-run noise (+-2 ms). The dispatch was
removed because dynamic dispatch inside an evaluation callback is a defect
whatever it costs, not because it was the bottleneck -- the bottleneck is the
AD sweep itself.
"""
function _hielow_gather!(b::AbstractVector{T}, x, idx) where {T}
    @inbounds for a in eachindex(b)
        b[a] = T(x[idx[a]])
    end
    return b
end

"""
    _hielow_hv!(Hv, H, v, idx, w) -> Hv

Scatter `w * H * v` into the full-length vector `Hv`, gathering `v` and
scattering the result through `idx`. Type-stable for the same reason as
[`_hielow_gather!`](@ref).
"""
function _hielow_hv!(Hv, H::AbstractMatrix{T}, v, idx, w::T) where {T}
    nc = size(H, 1)
    @inbounds for a in 1:nc
        s = zero(T)
        for b in 1:nc
            s += H[a, b] * T(v[idx[b]])
        end
        Hv[idx[a]] += w * s
    end
    return Hv
end

"""
    parkch_stratec_model(name; T = Float64, backend = nothing) -> ExaModel

Build PARKCH or STRATEC as an `ExaModel` whose objective callbacks are the
generic Julia port of `SUBROUTINE HIELOW` ([`hielow_loglik`](@ref)) rather
than the compiled Fortran the SIF file appends.

Structure (variables, bounds, start point, the single objective group and its
element's variable map) comes from `sif_decode(...; external = :defer)`; the
tree and the estimation sample come from [`hielow_parse`](@ref) reading the
`DATA` statements in the same file. No `sifdecoder` and no Fortran toolchain.

The gradient is one ForwardDiff sweep over the value function; the Hessian is
one full-chunk `ForwardDiff.hessian`, kept in the closure and keyed on the
point, so ExaModels' column-by-column Hessian reconstruction pays for it once.
See this file's header for the measured cost and for the two places where the
port's arithmetic deliberately differs from the Fortran's.

!!! warning "Second derivatives need the with-`y` NLPModels methods"
    As for AKIVA/HIELOW and the `BA-L*` family, ExaModels 0.12.1 routes a
    `ScalarNonlinearOracle`'s Hessian only through `hprod!(m, x, y, v, Hv)`
    and `hess_coord!(m, x, y, h)`. The no-`y` methods that
    `NLPModels.hprod(nlp, x, v)` and `NLPModels.hess(nlp, x)` use for an
    unconstrained model return ZERO. Pass `y = zeros(0)`. Upstream and
    pre-existing, not a property of this port.

`T = Float32` builds and evaluates (`obj(x0) = 2150.1487` for PARKCH against
the Float64 `2150.126124609186`, i.e. the accuracy a 1386-term
single-precision sum can deliver), and so does `T = BigFloat`, which is what
arbitrates a Float64 result.
"""
function parkch_stratec_model(name::AbstractString; T = Float64, backend = nothing)
    # The Hessian cache below is a mutable host array and `hielow_loglik`
    # indexes its scratch at a runtime node number, so these callbacks run on
    # the CPU; claiming a backend would misreport where the model runs.
    backend === nothing || error(
        "parkch_stratec_model's callbacks are host loops over hielow_loglik " *
        "and run on the CPU; use logit_kernel_model for a backend")
    name in ("PARKCH", "STRATEC") ||
        error("parkch_stratec_model handles PARKCH and STRATEC, not $name")

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
    # PARKCH maps B15 -> THE1 and STRATEC maps B10 -> THE1, which is also the
    # index the tree's THETA entries point at; `hielow_parse` already checked
    # THETA <= NCOEF, so a mismatch here would be a variable-map surprise.
    idx = ntuple(i -> e.varmap[Symbol(:B, i)], d.ncoef)
    length(unique(idx)) == d.ncoef ||
        error("$name: B1..B$(d.ncoef) do not map to distinct variables")

    n = length(P.varnames)
    nc = d.ncoef
    wT, cT = T(w), T(g.constant)

    loglik = b -> hielow_loglik(b, d)

    # Work buffers and the point-keyed Hessian cache. `bbuf` is the gathered
    # coefficient vector, `hpt` the point the cached `Hc` belongs to -- NaN so
    # the first call always misses.
    bbuf = zeros(T, nc)
    hpt = fill(T(NaN), nc)
    Hc = zeros(T, nc, nc)
    gcfg = ForwardDiff.GradientConfig(loglik, bbuf, ForwardDiff.Chunk{nc}())
    hcfg = ForwardDiff.HessianConfig(loglik, bbuf, ForwardDiff.Chunk{nc}())

    fbuf = zeros(T, nc)
    f = x -> wT * hielow_loglik(_hielow_gather!(fbuf, x, idx), d) - cT

    grad! = (gv, xv) -> begin
        fill!(gv, zero(eltype(gv)))
        gb = ForwardDiff.gradient(loglik, _hielow_gather!(bbuf, xv, idx), gcfg)
        for a in 1:nc
            gv[idx[a]] += wT * gb[a]
        end
        gv
    end

    # One AD pass per point, reused by every column ExaModels asks for.
    hess_at! = xv -> begin
        b = _hielow_gather!(bbuf, xv, idx)
        if b != hpt
            ForwardDiff.hessian!(Hc, loglik, b, hcfg)
            copyto!(hpt, b)
        end
        Hc
    end

    hvp! = (Hv, xv, v) -> begin
        fill!(Hv, zero(eltype(Hv)))
        _hielow_hv!(Hv, hess_at!(xv), v, idx, wT)
    end

    # Dense lower triangle of the coefficient block: 120 entries for PARKCH,
    # 55 for STRATEC. Every coefficient appears in every observation's utility
    # (through the nest scale if nowhere else), so the block really is dense.
    hr = [max(idx[a], idx[b]) for a in 1:nc for b in 1:a]
    hc = [min(idx[a], idx[b]) for a in 1:nc for b in 1:a]

    c = ExaModels.ExaCore(T)
    c, x = ExaModels.add_var(c, n; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    c = ExaModels.objective(c, ExaModels.ScalarNonlinearOracle(;
        nvar = n, f = f, grad! = grad!, hvp! = hvp!,
        hess_rows = hr, hess_cols = hc, adapt = Val(true)))
    return ExaModels.ExaModel(c)
end

"""
    parkch_model(; T = Float64, backend = nothing) -> ExaModel

PARKCH -- a 15-coefficient, 1386-observation three-level nested logit fitted to
a stated-preference parking survey in Namur -- as a native `ExaModel`. Its SIF
file is the largest in the collection (34670 lines, almost all of it the
estimation sample as Fortran `DATA`). See [`parkch_stratec_model`](@ref).
"""
parkch_model(; kwargs...) = parkch_stratec_model("PARKCH"; kwargs...)

"""
    stratec_model(; T = Float64, backend = nothing) -> ExaModel

STRATEC -- a 10-coefficient, 1573-observation nested logit over three nests
sharing one log-sum scale, fitted to a parking-choice study in Brussels -- as a
native `ExaModel`. Second largest SIF file in the collection (24339 lines). See
[`parkch_stratec_model`](@ref).
"""
stratec_model(; kwargs...) = parkch_stratec_model("STRATEC"; kwargs...)
