# RAYBENDS and BLEACHNG: generic Julia ports of their external Fortran.
#
# Two unrelated problems share this file because they share a shape: both are
# objective-only (no constraints), both have one external element type whose
# VALUE depends on the variables, and both therefore become a
# `ScalarNonlinearOracle` over a plain, type-generic Julia function with every
# derivative taken by ForwardDiff over that same function. Neither can be
# folded into the group's `lin` the way VANDANIUM's element can: a folded
# element value has to be a constant per element, and these are not.
#
# In both problems the SIF supplies hand-coded derivative routines -- RAYBENDS
# declares `FVAL`/`GVAL`/`HVAL`, BLEACHNG `FVAL`/`GVAL` -- and in both only
# `FVAL` is ported. The G/H routines are never read for their formulas; they
# are the error class that made the legacy HS67 port wrong in the fourth digit.
#
# ===========================================================================
# RAYBENDS -- what the Fortran computes
# ===========================================================================
#
# A ray crossing a 2D medium whose slowness varies with depth is a cubic
# beta-spline through NK+1 knots (X(i), Z(i)); the objective is the travel
# time, Fermat's principle. One group `TIME(i)` per spline segment, each
# holding one element `T(i)` of type `TT`, each group scaled by 2.0 (the SIF's
# 'SCALE' divides), so
#
#     obj(x) = sum_i FVAL(T(i)) / 2.
#
# `FVAL` -> `EVAL(DER=0, ...)` evaluates one segment with M = 10 subintervals.
# Writing q_i(t) for the four beta-spline basis functions (`Q(I,X)`) and
# t_k = 0.1 (k-1), k = 1..11:
#
#     z_k    = ZZ1 q1(t_k) + ZZ2 q2(t_k) + ZZ3 q3(t_k) + ZZ4 q4(t_k)
#     c_k    = 1 + 0.01 z_k                               (local wave speed)
#     dx_k   = X2MX1 (q2(t_{k+1})-q2(t_k)) + X3MX1 (...) + X4MX1 (...)
#     dz_k   = z_{k+1} - z_k
#     FVAL   = sum_{k=1..10} (1/c_k + 1/c_{k+1}) sqrt(dx_k^2 + dz_k^2)
#
# i.e. a trapezoidal slowness times the polyline length -- 2 x the travel time,
# which is what the group scale 2.0 divides out. `X2MX1 = X2-X1` etc. and
# `ZZi = Zi` are the element's SIF internal variables (the element is invariant
# under translation in X); this port works directly in the eight ELEMENTAL
# variables (X1..X4, Z1..Z4) and lets AD apply that linear map, so the SIF's
# `R` (range) rows need not be reproduced either. Note that the Fortran forms
# `dx` from the three differences only -- there is no `q1` term -- which is the
# exact statement of translation invariance and is ported as written.
#
# q_i(t_k) and their differences depend on nothing but t_k, so they are
# constants: a 4x11 table built once per model (`raybends_table`). What is left
# depending on the variables is the five lines above, ~15 operations x 10
# subintervals.
#
# THE PRECISION TRAP -- and here it bites. `Q` and `EVAL` have no `implicit
# none`, and five constants in the value path are written WITHOUT a `D0`
# exponent, so Fortran loads each as default REAL, i.e. SINGLE precision, and
# widens it:
#
#       Q = 0.166666667 - 0.5 * X + 0.5 * X * X - 0.1666667 * X**3
#       Q = 0.666666667 - X * X + 0.5 * X**3
#       C(K) = 1.0D0 + 0.01 * Z(K)
#
# 0.166666667, 0.666666667, 0.1666667 and 0.01 sit 2.8e-8, 2.9e-8, 8.6e-9 and
# 2.2e-8 relative from their Float32 neighbours (0.5 is exact in both
# readings). MEASURED against CUTEst at x0 and at a perturbed point, the data
# discriminates by eight orders of magnitude:
#
#                      obj                     grad
#   Float32 reading    0.0  (bit-identical)    2.9e-16, 2.2e-16
#   Float64 reading    7.3e-7, 4.1e-8          9.7e-7,  1.6e-6
#
# so the port rounds them through Float32 (`single = true`), and
# `single = false` keeps the other reading available for the record.
#
# Second derivatives ARE supplied here (the `INDIVIDUALS` block has a full `H`
# section through `HVAL`, and the classification `OXR2-MY-V-0` ends in `2`), so
# unlike SARO this port's Hessian is checked directly against CUTEst's: the
# FULL sparse Hessian agrees to 6.2e-14 and 1.5e-13 at the two points, over the
# same 15351 nonzeros CUTEst declares, and `hprod` to 1.3e-14 .. 1.4e-13.
#
# ===========================================================================
# BLEACHNG -- what the Fortran computes
# ===========================================================================
#
# Ozone mass transfer into water, fitted to four batch experiments. The state
# is the (gas-phase, liquid-phase) ozone concentration and obeys a LINEAR,
# constant-coefficient ODE (`Fsub`), with Q_out = Q_in:
#
#     Vg y1' = Qin Cin - y1 Qin - kLa (H Cin - y2) Vl
#        y2' = kLa (H Cin - y2)
#
# `FVAL` integrates it from t = 0 to the element's parameter `T` with ODESSA
# (ACM 658, an LSODE variant, MF = 21, RTOL = 1e-12, ATOL = 1e-14) and returns
# y1 or y2 according to the parameter `REALID`. Each observation contributes a
# group of type `L2` (`ALPHA**2`) whose constant is the measurement, so
#
#     obj(x) = sum over observations ( y_ID(T) - z )^2
#            + sum over the input-variable rows ( QIN(K) - z )^2
#                                              + ( CO3IN(K) - z )^2.
#
# SHAPE: the ODE IS SOLVED IN CLOSED FORM, not numerically. With
# a = Qin/Vg, k = kLa, L = H Cin, D = L - y2(0), c = k Vl D / Vg:
#
#     y2(t) = L + (y2(0) - L) e^{-kt}
#     y1(t) = Cin + (y1(0) - Cin) e^{-at} - c t e^{-at} phi((a-k) t),
#             phi(u) = expm1(u)/u, phi(0) = 1
#
# (a - k vanishes at Qin/Vg = kLa, which is inside the box, hence `phi` rather
# than the algebraically equal (e^{-kt} - e^{-at})/(a-k). `expm1(u)/u` is
# accurate for every u != 0 -- there is no cancellation to avoid -- so the
# four-term Taylor series is used only inside |u| < sqrt(eps(T)), where its
# truncation error is u^4/120 in the value, u^3/30 in dphi/du and u^2/10 in
# d2phi/du2, i.e. 8e-35, 3e-26 and 1e-17 at the Float64 switch: at or below
# rounding for all three, which is what makes AD through the switch safe.)
# Verified against a 2e6-step RK4 integration at six parameter sets
# including a = k exactly: 1e-15..1e-13 relative. So the port does not
# reimplement ODESSA; it replaces it, and the agreement with CUTEst is limited
# by ODESSA's own 1e-12 tolerance rather than by this code -- measured 4.5e-11
# and 7.7e-10 on `obj` at the two test points, which is the honest floor here
# and not a defect of the port.
#
# ONE UNRELATED DISCREPANCY, worth knowing before it is mistaken for a bug:
# CUTEst's x0 for QIN(K) is 0.66666666666999996, ours 0.66666666666666663.
# The SIF computes it as `A/ QIN(K) QIN(K) MIN/H` = 40/60, so ours is the
# correctly rounded quotient and CUTEst's start point carries about twelve
# significant digits -- a 3.3e-12 difference that comes from SIFDECODE's
# OUTSDIF.d, not from either model. Every comparison below evaluates both
# models at CUTEst's x0, so it does not affect any reported agreement.
#
# No precision trap in BLEACHNG's value path: `FVAL`, `Fsub` and ODESSA declare
# everything `DOUBLE PRECISION` and every literal in the value path carries a
# `D` exponent. The one undeclared variable, `MXSTEP = 10000`, starts with M,
# so Fortran's implicit rule makes it INTEGER, which is what it is used as.
# There is therefore no second reading to test here, as there is in RAYBENDS.
#
# THE REFERENCE'S GRADIENT W.R.T. QIN IS WRONG, by 39% at x0 and 60% at a
# perturbed point, and the port does NOT reproduce it. `DF` -- the
# inhomogeneity dF/dp of ODESSA's sensitivity equations, hand-written in the
# SIF -- supplies for the QIN column
#
#       DFDP( 1 ) = CO3IN / VOLGAS
#       DFDP( 2 ) = ZERO
#
# i.e. it differentiates the gas-phase balance with respect to Qin while
# holding Qout FIXED, even though `Fsub` two subroutines earlier sets
# `QOUT = QIN`; the term -y1/Vg is simply missing. (`JAC` is correct and the
# other four columns of `DF` are correct -- including the two that are
# identically zero because their parameter enters only through the initial
# condition, which is what `S01`/`S02` are for. Also worth noting that ODESSA
# is told NPAR = 5 with PAR = (kLa, QIN, CO3IN, VOLGAS, VOLLIQ) while `DF`'s
# five branches mean (kLa, CO3g0, CO3l0, QIN, CO3IN); the values are read by
# position and so are right, only the sensitivity labels are shifted.) With
# `JAC` correct, the broken column integrates to
#
#     S1 = (CO3IN/QIN)(1 - e^{-at}),  S2 = 0
#
# and MEASURED: that reconstruction reproduces the reference's gradient to
# 7.5e-10 and 2.6e-9 on the QIN components, while the true derivative differs
# from it by 39% / 60%. The other 13 components agree with this port to 1.4e-8
# and 4.8e-9, which is ODESSA's sensitivity accuracy (the SIF's own note: "the
# moderate accuracy of the gradient vector"). A central difference of the
# reference's OBJECTIVE -- which is correct -- decides between the two: it
# matches THIS PORT to 1.1e-5 / 3.1e-4 (finite-difference limited) and the
# reference's own gradient only to 0.39 / 0.60. So the port's gradient is the
# true gradient of the problem CUTEst defines, and the reference's is not.
#
# THE REFERENCE ALSO HAS NO SECOND DERIVATIVES, and not in the zero-Hessian
# way. The `INDIVIDUALS` block gives `F` and `A`/`G` but no `H`, while the
# GROUP type `L2` does supply `H = 2.0`; the classification `SBR1-RN-17-0` ends
# in `1`. CUTEst therefore returns the Gauss-Newton part only,
#
#     sum 2 grad(alpha) grad(alpha)' ,  the  sum 2 alpha hess(alpha)  term lost,
#
# built from the gradient above. That is nonzero and plausible -- the worst kind
# of wrong. MEASURED: `2 J'J` assembled from the port's Jacobian WITH the broken
# QIN column reproduces CUTEst's Hessian to 7.2e-12 and 2.8e-12, which
# identifies the mechanism exactly; with the true column it is off by 18% / 6%,
# and the true Hessian is off by 62% / 78%. This port's Hessian is checked
# instead against a central difference of the reference's own gradient on the
# 13 rows where that gradient is the true derivative (6.5e-8 .. 2.7e-6,
# finite-difference limited); the remaining 4x4 QIN-QIN block has no correct
# reference to check against, so it is checked against a central difference of
# this port's gradient (7e-10 .. 1.4e-7), which the reference's objective has
# already validated. `test/native_verify.jl` re-runs the comparison.
#
# ===========================================================================
# Both: why an oracle and not an expression
# ===========================================================================
#
# Neither value is written as an ExaModels expression. ExaModels nests a type
# parameter per operation; a ~30-deep chain did not finish compiling in an hour
# on a 57-variable problem (see src/native/BA.jl), and these are ~150 and ~20
# operations deep. As plain Julia functions they are generic in the element
# type (Float64/Float32/BigFloat all evaluate), allocation-free and branch-free
# on values apart from BLEACHNG's `phi` series switch, which is a removable
# singularity and not a selector. The consequence is the known upstream gap
# recorded in CLAUDE.md: `ScalarNonlinearOracle` takes `hvp!`, not an explicit
# `hess!`, so a full objective `hess_coord` costs one `hvp!` per Hessian column
# -- 2050 of them for RAYBENDS, measured at ~1 s for one `hess_coord`. `hprod`
# is one call and is unaffected. (The declared pattern must be DEDUPLICATED
# for that reconstruction to be right; see `_rb_hess_pattern`. Deduplicated,
# RAYBENDS declares exactly the 15351 nonzeros CUTEst does.)

# ===========================================================================
# RAYBENDS
# ===========================================================================

"The eight elemental variables of element type `TT`, in SIF declaration order."
const RAYBENDS_EVARS = (:X1, :X2, :X3, :X4, :Z1, :Z2, :Z3, :Z4)

"Number of subintervals per spline segment (`M` in the Fortran)."
const RAYBENDS_M = 10

"""
    RaybendsTable{S}

The constant part of one RAYBENDS segment: the four beta-spline basis
functions sampled at `t_k = 0.1 (k-1)`, `k = 1..11` (`q`), the three
differences `q_i(t_{k+1}) - q_i(t_k)` for `i = 2,3,4` that build `dx` (`dxd`),
and the slowness coefficient `cz` (the Fortran's `0.01`).

Element type `S` carries the working precision, so a Float32 model gets a
Float32 table and never silently evaluates in Float64.
"""
struct RaybendsTable{S}
    q::NTuple{11,NTuple{4,S}}
    dxd::NTuple{10,NTuple{3,S}}
    cz::S
end

# The four inexact bare (single precision) constants of `Q` and `EVAL`, in the
# two possible readings -- see the header. The fifth, `0.5`, is exact either
# way and needs no choice.
_rb_const(::Type{S}, c::Float64, single::Bool) where {S} =
    S(single ? Float64(Float32(c)) : c)

"""
    raybends_q(x, c1, c2, c3, half) -> NTuple{4,S}

`Q(I, X)` for `I = 1..4` at one point: the cubic beta-spline basis. Operation
order follows the Fortran statement by statement (`0.5 * X * X` is
`(0.5 X) X`, `X**3` is `(X X) X`), so a Float64 evaluation reproduces its bit
pattern and not merely its value.
"""
@inline function raybends_q(x::S, c1::S, c2::S, c3::S, half::S) where {S}
    x3 = (x * x) * x
    q1 = (c1 - half * x) + (half * x) * x - c3 * x3
    q2 = (c2 - x * x) + half * x3
    q3 = ((c1 + half * x) + (half * x) * x) - half * x3
    q4 = c1 * x3
    return (q1, q2, q3, q4)
end

"""
    raybends_table(::Type{S}; single = true) -> RaybendsTable{S}

Build the constant table above. `single = true` rounds the Fortran's four bare
decimal constants through Float32, which is what Fortran's implicit typing
does to them and what reproduces CUTEst (see the header); `single = false`
reads them as Float64.
"""
function raybends_table(::Type{S}; single::Bool = true) where {S}
    c1 = _rb_const(S, 0.166666667, single)
    c2 = _rb_const(S, 0.666666667, single)
    c3 = _rb_const(S, 0.1666667, single)
    cz = _rb_const(S, 0.01, single)
    half = S(0.5)
    # T(K) = 0.1D0 * ( K - 1.0D0 )
    tk = ntuple(k -> S(1) / S(10) * S(k - 1), Val(RAYBENDS_M + 1))
    q = ntuple(k -> raybends_q(tk[k], c1, c2, c3, half), Val(RAYBENDS_M + 1))
    dxd = ntuple(k -> (q[k + 1][2] - q[k][2],
                       q[k + 1][3] - q[k][3],
                       q[k + 1][4] - q[k][4]), Val(RAYBENDS_M))
    return RaybendsTable{S}(q, dxd, cz)
end

"""
    raybends_time(v, tab) -> T

`FVAL` for one spline segment: twice the travel time along it, from the eight
elemental variables `v = (X1, X2, X3, X4, Z1, Z2, Z3, Z4)`.

Generic in the element type, allocation-free, branch-free on values, and
unrolled over the ten subintervals so nothing is indexed at run time -- which
is what keeps it callable from a GPU kernel and keeps nested duals in
registers.
"""
@inline function raybends_time(v::NTuple{8,T}, tab::RaybendsTable{S}) where {T,S}
    x1, x2, x3, x4, z1, z2, z3, z4 = v
    # the element's SIF internal variables; X enters only through differences
    u2 = x2 - x1
    u3 = x3 - x1
    u4 = x4 - x1
    cz = tab.cz
    one_ = one(T)

    q0 = tab.q[1]
    zk = z1 * q0[1] + z2 * q0[2] + z3 * q0[3] + z4 * q0[4]
    ck = one_ + cz * zk
    f = zero(T)
    Base.Cartesian.@nexprs 10 k -> begin
        qn = tab.q[k + 1]
        zn = z1 * qn[1] + z2 * qn[2] + z3 * qn[3] + z4 * qn[4]
        cn = one_ + cz * zn
        d = tab.dxd[k]
        dx = u2 * d[1] + u3 * d[2] + u4 * d[3]
        dz = zn - zk
        f += (inv(ck) + inv(cn)) * sqrt(dx * dx + dz * dz)
        zk = zn
        ck = cn
    end
    return f
end

"Gradient of [`raybends_time`](@ref) w.r.t. its eight elemental variables."
@inline function raybends_grad(v::NTuple{8,T}, tab::RaybendsTable) where {T}
    D = ForwardDiff.Dual{:RAYBENDS,T,8}
    vd = ntuple(i -> D(v[i],
                       ForwardDiff.Partials(ntuple(j -> ifelse(i == j, one(T), zero(T)),
                                                   Val(8)))),
                Val(8))
    res = raybends_time(vd, tab)
    return ntuple(i -> ForwardDiff.partials(res, i), Val(8))
end

"""
    raybends_hvp(v, d, tab) -> NTuple{8,T}

`hess(raybends_time)(v) * d` for one segment, as the directional derivative of
the gradient: one 8-partial sweep over one 1-partial sweep, which is ~4x
cheaper than forming the 8x8 Hessian and is all the oracle's `hvp!` needs.
"""
@inline function raybends_hvp(v::NTuple{8,T}, d::NTuple{8,T},
                              tab::RaybendsTable) where {T}
    O = ForwardDiff.Dual{:RAYBENDSdir,T,1}
    vo = ntuple(i -> O(v[i], ForwardDiff.Partials((d[i],))), Val(8))
    g = raybends_grad(vo, tab)
    return ntuple(i -> ForwardDiff.partials(g[i], 1), Val(8))
end

"One SIF segment: where its eight elemental variables live, and its weight."
struct RaybendsElem
    idx::NTuple{8,Int}
    w::Float64
end

"""
    raybends_elements(P) -> Vector{RaybendsElem}

Read the segments out of a `sif_decode(...; external = :defer)` problem.

Every objective group must hold exactly one `TT` element and nothing else; the
group's `'SCALE'` divides the weight, which is where the factor 1/2 comes from.
Anything else means the SIF changed and is an error rather than a silent
approximation. Note the variable map is NOT injective -- `T(0)` has
`X1 = X2 = X3 = X(0)` -- so gradients must be scattered with `+=`, which is
exactly the chain rule for a repeated argument.
"""
function raybends_elements(P)
    els = RaybendsElem[]
    for gn in P.grouporder
        g = P.groups[gn]
        g.kind == 'N' || error("RAYBENDS: group $gn is a constraint; expected none")
        isempty(g.lin) || error("RAYBENDS: group $gn has linear terms")
        g.constant == 0.0 || error("RAYBENDS: group $gn has constant $(g.constant)")
        ExaModelsSIF.gtypename(P, g) === nothing ||
            error("RAYBENDS: group $gn has a group type; expected the trivial one")
        length(g.elems) == 1 ||
            error("RAYBENDS: group $gn holds $(length(g.elems)) elements; expected 1")
        en, w = g.elems[1]
        e = P.elements[en]
        et = ExaModelsSIF.eltypeof(P, e).name
        et == "TT" || error("RAYBENDS: element $en has type $et; expected TT")
        g.scale > 0.0 || error("RAYBENDS: group $gn has non-positive scale $(g.scale)")
        push!(els, RaybendsElem(ntuple(a -> e.varmap[RAYBENDS_EVARS[a]], Val(8)),
                                w / g.scale))
    end
    return els
end

# ===========================================================================
# BLEACHNG
# ===========================================================================

"The five elemental variables of element type `A`, in SIF declaration order."
const BLEACHNG_EVARS = (:kLa, :CO3g0, :CO3l0, :QIN, :CO3IN)

"""
    bleachng_phi(u) -> T

`expm1(u)/u`, continued by its Taylor series to `phi(0) = 1`.

The removable singularity is real: it sits at `QIN/VOLGAS = kLa`, inside the
box the solver searches. `expm1(u)/u` is accurate wherever it is defined, so
the series is used only inside `|u| < sqrt(eps(T))`, where its truncation
error in the value and in the first two derivatives is at or below rounding --
which is what makes AD through the switch safe. The threshold follows the
working precision (`eps` of a `ForwardDiff.Dual` is `eps` of its value type).
"""
@inline function bleachng_phi(u::T) where {T}
    small = u * u < eps(T)
    s = one(T) + u * (one(T) / T(2) + u * (one(T) / T(6) + u * (one(T) / T(24))))
    return ifelse(small, s, expm1(u) / u)
end

"""
    bleachng_conc(v, p) -> T

`FVAL` for one observation: the ozone concentration the model predicts at the
element's observation time.

`v = (kLa, CO3g0, CO3l0, QIN, CO3IN)` are the five elemental variables and
`p = (VOLGAS, VOLLIQ, H, T, SEL)` the element parameters that matter, with
`SEL = REALID - 1`, i.e. 0 for the gas phase and 1 for the liquid phase. `SEL`
is a blend coefficient rather than a branch, so the function stays
straight-line code.

This is the CLOSED-FORM solution of the linear ODE that the Fortran integrates
with ODESSA -- see the header for the derivation and for what that costs in
agreement.
"""
@inline function bleachng_conc(v::NTuple{5,T}, p::NTuple{5,S}) where {T,S}
    kla, cg0, cl0, qin, cin = v
    volgas, volliq, h, t, sel = p

    a = qin / volgas                 # Qout = Qin, so the gas washout rate
    l = h * cin                      # liquid-phase saturation level
    ekt = exp(-kla * t)
    y2 = l + (cl0 - l) * ekt

    d = l - cl0
    cc = kla * volliq * d / volgas
    eat = exp(-a * t)
    y1 = cin + (cg0 - cin) * eat - cc * t * eat * bleachng_phi((a - kla) * t)

    return y1 + sel * (y2 - y1)
end

"Gradient of [`bleachng_conc`](@ref) w.r.t. its five elemental variables."
@inline function bleachng_grad(v::NTuple{5,T}, p::NTuple{5,S}) where {T,S}
    D = ForwardDiff.Dual{:BLEACHNG,T,5}
    vd = ntuple(i -> D(v[i],
                       ForwardDiff.Partials(ntuple(j -> ifelse(i == j, one(T), zero(T)),
                                                   Val(5)))),
                Val(5))
    out = bleachng_conc(vd, p)
    return ForwardDiff.value(out), ntuple(i -> ForwardDiff.partials(out, i), Val(5))
end

"""
    bleachng_grad_hvp(v, d, p) -> (value, gradient, hess * d)

Value, gradient and one Hessian-vector product of [`bleachng_conc`](@ref) in a
single nested sweep (5 inner partials over 1 outer partial). The oracle's
`hvp!` needs all three, because the term it differentiates is `(conc - z)^2`.
"""
@inline function bleachng_grad_hvp(v::NTuple{5,T}, d::NTuple{5,T},
                                   p::NTuple{5,S}) where {T,S}
    O = ForwardDiff.Dual{:BLEACHNGdir,T,1}
    vo = ntuple(i -> O(v[i], ForwardDiff.Partials((d[i],))), Val(5))
    f, g = bleachng_grad(vo, p)
    return ForwardDiff.value(f),
           ntuple(i -> ForwardDiff.value(g[i]), Val(5)),
           ntuple(i -> ForwardDiff.partials(g[i], 1), Val(5))
end

"""
    BleachngObs

One observation group: where its five elemental variables live, the element
parameters `(VOLGAS, VOLLIQ, H, T, REALID-1)`, the element weight divided by
the group scale, and the measurement that the group's constant carries.
"""
struct BleachngObs{S}
    idx::NTuple{5,Int}
    par::NTuple{5,S}
    w::S
    z::S
end

"""
    bleachng_groups(P, ::Type{S}) -> (Vector{BleachngObs{S}}, Vector{Tuple{S,Int,S}})

Split a `sif_decode(...; external = :defer)` problem into the groups that hold
an `A` element and the groups that are purely linear.

The second return value is `(a, j, z)` per linear group -- the linear
coefficient, the variable it multiplies and the group constant -- which stay a
normal SIMD ExaModels objective (`(a x_j - z)^2`); only the element groups need
the oracle. Every group must be of type `L2` with unit scale, hold at most one
element, and not mix an element with linear terms; otherwise the SIF changed.
"""
function bleachng_groups(P, ::Type{S}) where {S}
    obs = BleachngObs{S}[]
    lins = Tuple{S,Int,S}[]
    for gn in P.grouporder
        g = P.groups[gn]
        g.kind == 'N' || error("BLEACHNG: group $gn is a constraint; expected none")
        ExaModelsSIF.gtypename(P, g) == "L2" ||
            error("BLEACHNG: group $gn has group type " *
                  "$(ExaModelsSIF.gtypename(P, g)); expected L2")
        g.scale == 1.0 ||
            error("BLEACHNG: group $gn is scaled by $(g.scale); only 1 is handled")
        if isempty(g.elems)
            length(g.lin) == 1 ||
                error("BLEACHNG: linear group $gn has $(length(g.lin)) terms; expected 1")
            j, a = g.lin[1]
            push!(lins, (S(a), j, S(g.constant)))
        else
            length(g.elems) == 1 ||
                error("BLEACHNG: group $gn holds $(length(g.elems)) elements; expected 1")
            isempty(g.lin) ||
                error("BLEACHNG: group $gn mixes an element with linear terms")
            en, w = g.elems[1]
            e = P.elements[en]
            et = ExaModelsSIF.eltypeof(P, e).name
            et == "A" || error("BLEACHNG: element $en has type $et; expected A")
            id = e.parvals[:REALID]
            (id == 1.0 || id == 2.0) ||
                error("BLEACHNG: element $en has REALID = $id; expected 1 or 2")
            push!(obs, BleachngObs{S}(ntuple(a -> e.varmap[BLEACHNG_EVARS[a]], Val(5)),
                                      (S(e.parvals[:VOLGAS]), S(e.parvals[:VOLLIQ]),
                                       S(e.parvals[:H]), S(e.parvals[:T]), S(id - 1)),
                                      S(w), S(g.constant)))
        end
    end
    return obs, lins
end

# ===========================================================================
# Assembly
# ===========================================================================

"""
    _rb_hess_pattern(blocks, k) -> (rows, cols)

The deduplicated lower-triangular Hessian pattern of a set of `k`-variable
blocks.

Deduplication is NOT cosmetic. `ExaModels.hess_coord!` fills a scalar oracle's
Hessian by writing one value per declared nonzero, so a coordinate declared
twice is written twice and SUMMED by the assembler -- i.e. counted twice. (The
`VectorNonlinearOracle` path, which calls `hess!` and accumulates, is where
duplicates are harmless.) Both problems here have overlapping blocks:
RAYBENDS' segments share knots and BLEACHNG's observations all share `kLa`.
"""
function _rb_hess_pattern(blocks, k::Int)
    seen = Set{Tuple{Int,Int}}()
    rows = Int[]
    cols = Int[]
    for idx in blocks, a in 1:k, b in 1:a
        i, j = idx[a], idx[b]
        p = (max(i, j), min(i, j))
        p in seen && continue
        push!(seen, p)
        push!(rows, p[1])
        push!(cols, p[2])
    end
    return rows, cols
end

"""
    _raybends_decode() -> (P, els)

Decode RAYBENDS, check that it has the shape every builder here assumes, and
read out one [`RaybendsElem`](@ref) per spline segment.
"""
function _raybends_decode()
    P = ExaModelsSIF.sif_decode(sif_problem_path("RAYBENDS"); external = :defer)
    for (tname, fn) in P.external_types
        (tname == "TT" && fn === :FVAL) ||
            error("RAYBENDS: expected deferred TT calling FVAL, found $tname => $fn")
    end
    all(==(1.0), P.varscale) || error("RAYBENDS: variable scaling is not supported here")
    els = raybends_elements(P)
    isempty(els) && error("RAYBENDS: no TT elements found")
    return P, els
end

"""
    _bleachng_decode(::Type{S}) -> (P, obs, lins)

Decode BLEACHNG, check that it has the shape every builder here assumes, and
split its groups into the observations that hold an `A` element and the purely
linear ones.
"""
function _bleachng_decode(::Type{S}) where {S}
    P = ExaModelsSIF.sif_decode(sif_problem_path("BLEACHNG"); external = :defer)
    for (tname, fn) in P.external_types
        (tname == "A" && fn === :FVAL) ||
            error("BLEACHNG: expected deferred A calling FVAL, found $tname => $fn")
    end
    all(==(1.0), P.varscale) || error("BLEACHNG: variable scaling is not supported here")
    obs, lins = bleachng_groups(P, S)
    isempty(obs) && error("BLEACHNG: no A elements found")
    return P, obs, lins
end

function _rb_backend_check(fname::AbstractString, backend)
    backend === nothing || error(
        "$fname does not take a backend: the port's callbacks run on the " *
        "CPU. The value functions are allocation- and branch-free and so are " *
        "GPU-safe; ignoring `backend` would misreport the model as being on " *
        "the device.")
end

"""
    raybends_model(; T = Float64, backend = nothing, single = true) -> ExaModel

Build RAYBENDS as an ExaModel whose objective is the generic Julia port of the
SIF's `FVAL` rather than compiled Fortran.

One `ScalarNonlinearOracle` holds all 1026 spline segments; its value,
gradient and Hessian-vector product come from ForwardDiff over
[`raybends_time`](@ref), so the SIF's hand-coded `GVAL`/`HVAL` are never used.
Variables are in SIF declaration order (`sif_variable_names("RAYBENDS")`),
which is NOT CUTEst's order.

`single = false` switches the four inexact bare Fortran constants from their
Float32 reading to a Float64 one; it is wrong against CUTEst by 7.3e-7 on
`obj` at x0 and exists only to document the measurement (see the header).

Needs no Fortran toolchain and no `sifdecoder`: the structure comes from
`sif_decode(...; external = :defer)`.

`T` may be `Float32` or `BigFloat` as well as `Float64`: the callbacks are
generic in the element type and ExaModels allocates the oracle's work buffers
at it, so a non-Float64 model carries its own precision rather than
round-tripping through double. `hess_coord` at `BigFloat` is the one callback
to avoid here -- this oracle has no explicit `hess!`, so ExaModels
reconstructs the Hessian with one `hvp!` per variable, 2050 passes over all
1026 segments under nested `BigFloat` duals. `raybends_kernel_model` supplies
an explicit `hess!` and has no such cost.
"""
function raybends_model(; T = Float64, backend = nothing, single::Bool = true)
    _rb_backend_check("raybends_model", backend)
    P, els = _raybends_decode()
    tab = raybends_table(T; single)
    n = length(P.varnames)
    hr, hc = _rb_hess_pattern((e.idx for e in els), 8)

    f = function (xv)
        E = eltype(xv)
        s = zero(E)
        for e in els
            s += E(e.w) * raybends_time(ntuple(a -> xv[e.idx[a]], Val(8)), tab)
        end
        return s
    end
    grad! = function (g, xv)
        E = eltype(g)
        fill!(g, zero(E))
        for e in els
            gr = raybends_grad(ntuple(a -> E(xv[e.idx[a]]), Val(8)), tab)
            w = E(e.w)
            for a in 1:8
                g[e.idx[a]] += w * gr[a]
            end
        end
        return g
    end
    hvp! = function (Hv, xv, v)
        E = eltype(Hv)
        fill!(Hv, zero(E))
        for e in els
            hd = raybends_hvp(ntuple(a -> E(xv[e.idx[a]]), Val(8)),
                              ntuple(a -> E(v[e.idx[a]]), Val(8)), tab)
            w = E(e.w)
            for a in 1:8
                Hv[e.idx[a]] += w * hd[a]
            end
        end
        return Hv
    end

    c = ExaModels.ExaCore(T)
    c, _ = ExaModels.add_var(c, n; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    c = ExaModels.objective(c, ExaModels.ScalarNonlinearOracle(;
        nvar = n, f = f, grad! = grad!, hvp! = hvp!,
        hess_rows = hr, hess_cols = hc, adapt = Val(true)))
    return ExaModels.ExaModel(c)
end

"""
    bleachng_model(; T = Float64, backend = nothing) -> ExaModel

Build BLEACHNG as an ExaModel whose objective is the generic Julia port of the
SIF's `FVAL` -- the closed-form solution of the ODE that the appended ODESSA
integrates numerically.

The 1602 observation groups go into one `ScalarNonlinearOracle` over
[`bleachng_conc`](@ref); the 1602 purely linear groups (`QIN(K)` and
`CO3IN(K)` against their measured values) stay a normal SIMD ExaModels
objective, since they contain no external function. Derivatives come from
ForwardDiff, so the SIF's `GVAL` (ODESSA's sensitivity equations) is not used
-- and this port's Hessian is the true one where CUTEst's is Gauss-Newton; see
the header.

Variables are in SIF declaration order. `T` may be `Float32` or `BigFloat` as
well as `Float64`, as in [`raybends_model`](@ref); with 17 variables the
Hessian reconstruction that costs RAYBENDS dearly is cheap here.
"""
function bleachng_model(; T = Float64, backend = nothing)
    _rb_backend_check("bleachng_model", backend)
    P, obs, lins = _bleachng_decode(T)
    n = length(P.varnames)
    hr, hc = _rb_hess_pattern((o.idx for o in obs), 5)

    f = function (xv)
        E = eltype(xv)
        s = zero(E)
        for o in obs
            r = E(o.w) * bleachng_conc(ntuple(a -> xv[o.idx[a]], Val(5)), o.par) - E(o.z)
            s += r * r
        end
        return s
    end
    grad! = function (g, xv)
        E = eltype(g)
        fill!(g, zero(E))
        for o in obs
            val, gr = bleachng_grad(ntuple(a -> E(xv[o.idx[a]]), Val(5)), o.par)
            w = E(o.w)
            r2 = 2 * (w * val - E(o.z)) * w
            for a in 1:5
                g[o.idx[a]] += r2 * gr[a]
            end
        end
        return g
    end
    # hess((w conc - z)^2) = 2 w^2 grad grad' + 2 (w conc - z) w hess(conc)
    hvp! = function (Hv, xv, v)
        E = eltype(Hv)
        fill!(Hv, zero(E))
        for o in obs
            d = ntuple(a -> E(v[o.idx[a]]), Val(5))
            val, gr, hd = bleachng_grad_hvp(ntuple(a -> E(xv[o.idx[a]]), Val(5)), d, o.par)
            w = E(o.w)
            r2 = 2 * (w * val - E(o.z)) * w
            gv = zero(E)
            for a in 1:5
                gv += gr[a] * d[a]
            end
            c1 = 2 * w * w * gv
            for a in 1:5
                Hv[o.idx[a]] += c1 * gr[a] + r2 * hd[a]
            end
        end
        return Hv
    end

    c = ExaModels.ExaCore(T)
    c, x = ExaModels.add_var(c, n; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    if !isempty(lins)
        data = [(a = l[1], j = l[2], z = l[3]) for l in lins]
        c, _ = ExaModels.add_obj(c, Base.Generator(d -> (d.a * x[d.j] - d.z)^2, data))
    end
    c = ExaModels.objective(c, ExaModels.ScalarNonlinearOracle(;
        nvar = n, f = f, grad! = grad!, hvp! = hvp!,
        hess_rows = hr, hess_cols = hc, adapt = Val(true)))
    return ExaModels.ExaModel(c)
end
