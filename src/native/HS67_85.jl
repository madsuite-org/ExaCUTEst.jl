# HS67 and HS85: generic Julia ports of their external Fortran.
#
# Both SIF files append ONE Fortran routine that computes a whole cascade of
# intermediate quantities, and both declare one SIF element type per quantity,
# all of them over the same handful of problem variables. So one call of the
# routine produces every element value the problem needs, which is exactly the
# shape these ports take: a single generic Julia function returning a tuple of
# element values, interfaced to ExaModels as an external oracle, with every
# derivative coming from ForwardDiff over it. The Fortran's hand-coded G/H
# arrays are NOT ported -- that is the error class to stay clear of.
#
# SHAPE, and why it is not an ExaModels expression. ExaModels nests a type
# parameter per operation; a ~30-operation chain did not finish compiling in an
# hour on a 57-variable problem (see src/native/BA.jl). HS85's cascade is ~80
# operations and HS67's contains a FIXED-POINT ITERATION, which has no
# expression form at all. Neither can be folded either (VANDANIUM-style),
# because both depend on the variables rather than only on element parameters.
#
# ===========================================================================
# HS67 -- the iteration is the function, not the fixed point
# ===========================================================================
#
# WHAT THE FORTRAN COMPUTES. `HS67(X1, X2, X3, Y, G, H)` fills Y(2..8), the
# eight Hock-Schittkowski intermediate quantities of the Colville refinery
# model. Two of them, Y2 and Y4, are defined only implicitly, and the Fortran
# resolves each by a FIXED-POINT ITERATION with a loose stopping rule:
#
#       Y2 = 1.6 * X1                                   <- first guess
#   10  Y3 = 1.22 * Y2 - X1
#       Y6 = ( X2 + Y3 ) / X1
#       Y2C = 0.01 * X1 * ( 112 + 13.167*Y6 - 0.6667*Y6**2 )
#       IF ( ABS( Y2C - Y2 ) .GT. 0.001 ) THEN  Y2 = Y2C ; GO TO 10
#
#       Y4 = 93.0                                       <- first guess
#   40  Y5 = 86.35 + 1.098*Y6 - 0.038*Y6**2 + 0.325*( Y4 - 89 )
#       Y8 = 3*Y5 - 133
#       Y7 = 35.82 - 0.222*Y8
#       Y4C = 98000*X3 / ( Y2*Y7 + 1000*X3 )
#       IF ( ABS( Y4C - Y4 ) .GT. 0.001 ) THEN  Y4 = Y4C ; GO TO 40
#
# Note what the stopping rule does: when the test FAILS the loop exits WITHOUT
# taking the last step, so the Y2 (resp. Y4) that leaves the routine is the
# iterate BEFORE the final update, while Y3, Y6 (resp. Y5, Y7, Y8) are the
# values derived from it. And the derivative arrays G/H are propagated through
# each iteration step, i.e. they are the derivatives of the finite iteration
# map, not of the exact fixed point.
#
# THIS IS WHY THE PREVIOUS PORT WAS WRONG. Both relations are quadratics that
# can be solved in closed form, and a port that did so -- reasoning that the
# iteration converges to that root anyway -- produced -868.6357 for the
# objective at x0 where the reference is -868.7250, wrong in the FOURTH digit,
# with the gradient off by 4.4e-3 relative. It is not a rounding artifact: the
# tolerance is 0.001 in ABSOLUTE terms on quantities of size ~3000 and ~90, so
# the iteration stops a long way from the root (measured at x0: five Y2 steps
# ending 2.2e-4 from the previous iterate, twelve Y4 steps ending 7.4e-4
# away). The reference's value IS the finite iteration's value. So this port
# reproduces the iteration itself -- same first guess, same update, same
# stopping rule -- and lets ForwardDiff differentiate the iterates, which is
# precisely what the Fortran's G/H recursions do by hand. (Checked: the
# Fortran's G(2,1) is `0.01*F + X1*0.13167*G(6,1) - X1*0.013334*Y6*G(6,1)`,
# which is the exact chain rule for one step of the update, with
# 0.13167 = 0.01*13.167 and 0.013334 = 0.01*2*0.6667.)
#
# THE PRECISION FINDING -- and here it is decisive. `HS67` has no
# `implicit none`, and while every VARIABLE is declared DOUBLE PRECISION,
# every CONSTANT is written bare: `1.6`, `1.22`, `0.01`, `112.0`, `13.167`,
# `0.6667`, `0.001`, `93.0`, `86.35`, `1.098`, `0.038`, `0.325`, `89.0`,
# `3.0`, `133.0`, `35.82`, `0.222`, `1000.0`, `98000.0`. A bare Fortran
# literal is default REAL, i.e. SINGLE precision, and is only widened to
# double when it meets a double operand. Measured at x0 against the reference
# objective -868.7250146472379:
#
#   literals read as Float64   -868.7256519588518    off by 6.4e-04 (7.3e-07 rel)
#   literals rounded to Float32 -868.7250146472379   EXACT, all 16 digits
#
# So the single-precision reading is the right one and it is not a cosmetic
# difference -- it is a thousand times larger than Float64 rounding. Both are
# available through `single_literals`; the default is the one that matches.
#
# ===========================================================================
# HS85 -- one long cascade, no iteration, no precision trap
# ===========================================================================
#
# WHAT THE FORTRAN COMPUTES. `IFUN85(EVAL, X1..X5, CF,..., YF,...)` evaluates
# the wood-pulp plant model of Hock & Schittkowski 85: twenty quantities
# Y(1..20) and nineteen C(1..19), in one interleaved cascade (Y1, C1, Y2, C2,
# C3, Y3, Y4, C4, ... C19, Y18, Y19, Y20). The SIF declares two element types
# `Y` and `C` carrying an element parameter `PI = I + 0.01`, recovers the index
# as `I = INT(PI)`, and takes the value from `YF(I)` / `CF(I)` -- the first
# array argument in each half of the call list, i.e. the dummy arguments `Y`
# and `C`. (The remaining 40 arrays are the hand-coded gradient and Hessian,
# not ported.) `IFUN85` is an INTEGER FUNCTION whose result is dropped into an
# unused `IFLAG`, so the port returns only the values.
#
# NO PRECISION TRAP HERE, checked rather than assumed: every identifier
# assigned in `IFUN85` is declared (`A, B, D, E, F` DOUBLE PRECISION, plus the
# arrays and `TWO = 2.0D+0`), and every literal in the file carries a `D`
# exponent -- `grep` for a decimal constant without one finds nothing but
# comment lines. So Float64 throughout, which is what the measurement
# confirms.
#
# UNUSED ELEMENTS. `ELEMENT USES` declares C(1..19) and Y(1..20) but
# `GROUP USES` references only Y(1..20), C18 and C19; C(1..17) are
# intermediates that never reach a group. They are still computed, because the
# cascade needs them, but no row or objective term reads them.
#
# ===========================================================================
# ASSEMBLY (shared)
# ===========================================================================
#
# Every element of either problem is a function of ALL the problem's
# variables, and each row is (one element) + (linear terms) - (group
# constant). So the two models share one assembler -- in fact two, one per
# builder: `_hs_oracle_model` interfaces the value cascades through host loops,
# and `_hs_kernel_model` (end of this file) interfaces the same cascades
# through KernelAbstractions kernels and so accepts a `backend`. The host loops
# are what was verified against the Fortran, and the kernels are verified
# against them. Both assemble the same shape:
#
#   * the 14 (HS67) / 21 (HS85) rows go into a single `VectorNonlinearOracle`
#     in SIF group order, so row i is `sif_constraint_names(name)[i]`;
#   * the objective goes into a `ScalarNonlinearOracle`, linear terms and the
#     group constant included, because mixing an expression objective with an
#     oracle objective is not a shape ExaModels promises;
#   * the Jacobian is dense (n entries per row that holds an element) and the
#     Hessian sparsity is ONE lower triangle of the n x n block. Every row
#     touches the same variables, so the constraint Hessian is assembled by
#     summing the multipliers into one weight per element slot and taking a
#     single Hessian of that weighted sum -- one AD call per `hess!`, not one
#     per row.
#
# Structural zeros are left in: Y2, Y3 and Y6 of HS67 do not depend on X3, and
# most of HS85's elements depend on fewer than five variables, but padding to
# a dense block costs nothing at n = 3 or 5 and keeps the assembly honest
# about what it did not try to prove.
#
# WHAT IS NOT BRANCH-FREE. `hs67_elements` contains the two `while` loops, and
# their trip count depends on the values -- that is unavoidable, it IS the
# function the reference computes. The loops are bounded (`maxit`, default 64,
# against the 5 and 12 steps x0 needs) so the function always terminates; past
# the bound it returns the last iterate rather than raising, because raising
# from a GPU kernel is not an option and the Fortran itself would simply spin.
# Everything else in both ports is straight-line and allocation-free.
#
# ===========================================================================
# WHAT WAS MEASURED, INCLUDING WHAT DOES NOT MATCH
# ===========================================================================
#
# Against real `CUTEstModel`s at x0 and a perturbed point, permuted by SIF
# name; `test/native_verify.jl` re-runs it. Both problems are classified
# `OOI2-`, so the reference does supply second derivatives and the Hessian
# comparison is meaningful. Worst relative errors over both points:
#
#            obj            cons   grad     jac      hess      hprod
#   HS67     bit-equal      0      4.8e-08  1.7e-08  2.6e-10   2.5e-10
#   HS85     bit-equal      0      3.4e-18  8.0e-17  5.4e-03   8.4e-03
#
# VALUES AGREE EXACTLY, derivatives do not, and in BOTH cases the remaining
# gap is a defect in the SIF's hand-coded derivatives rather than in the port.
# Diagnosed, not assumed:
#
# HS67, the ~5e-8 gradient gap. The Fortran hand-codes its derivative
# constants as SEPARATE bare literals -- `0.13167` and `0.013334` where the
# chain rule on the value expression gives `0.01*13.167` and `0.01*2*0.6667`.
# In single precision those are different numbers:
#   Float32(0.13167)  = 0.13166999816894531
#   Float32(0.01)*Float32(13.167) = 0.13166999522589151   (2.2e-08 apart)
#   Float32(0.013334) vs Float32(0.01)*2*Float32(0.6667)  (2.0e-08 apart)
# A diagnostic transcription of the Fortran's own G recursion reproduces
# CUTEst's gradient BIT FOR BIT, which identifies the mechanism. A BigFloat
# central difference of this port's own (bit-exact) objective then settles
# which one is the derivative of the value: our AD agrees with it to 1.1e-14
# and CUTEst's to only 3.1e-07. So the port's gradient is right and the
# reference's is internally inconsistent with its own value at the 1e-7 level.
# There is no way to close the gap without porting the hand-coded G, which
# would mean shipping that inconsistency. (The Hessian gap, 2.6e-10, is the
# same mechanism further down the chain; `0.076 == 2*0.038` and
# `196000 == 2*98000` hold exactly in Float32, which is why it is smaller.)
#
# HS85, the ~5e-3 Hessian gap: TWO DROPPED MINUS SIGNS in the SIF's hand-coded
# second derivatives.
#   Y23( 15 ) = + Y2( 13 ) * C3( 13 ) / C( 13 )**2 - ...   should be  -
#   Y12( 12 ) = + C1( 11 ) * C2( 12 ) / C( 12 )**2 - ...   should be  -
# `Y(15) = Y(13)/C(13)` and `Y(12) = C(10)*X1 + C(11)/C(12)` need those terms
# negative; the other product-rule terms they omit are all identically zero
# here, so the sign is the whole error. Predicting the discrepancy as
# `2*Y2(13)*C3(13)/C(13)^2` and `2*C1(11)*C2(12)/C(12)^2` reproduces the
# measured difference to 1.6e-22 and 1.7e-21, at exactly those two entries and
# nowhere else. `Y(16) = A + B*Y(15) + D*Y(13) + E*Y(15)*Y(13)` inherits the
# Y15 error amplified by `B + E*Y(13)`, predicted and measured equal to the
# last bit -- which is why CON16 is the loudest row, and why the objective
# (which reads Y12 and Y16) is off by 6.4e-08. The other 17 element Hessians
# agree with ours to <= 1.4e-17.
# Independent confirmation that ours is the right one: a central difference of
# CUTEst's OWN gradient + Jacobian (first derivatives only, which the SIF gets
# right) matches our hprod to <= 2.5e-11 for every row and the objective,
# while CUTEst's own hprod misses it by up to 5.8e-03.
#
# So: neither derivative gap is closable from this side, and neither should be
# closed -- matching them would mean reproducing a bug. Both are reported
# rather than smoothed over.

# ---------------------------------------------------------------------------
# HS67: the value cascade

"""
    hs67_consts(::Type{S}; single_literals = true) -> NTuple{19,S}

The nineteen bare constants of the `HS67` Fortran, in the order
`(1.6, 1.22, 0.01, 112, 13.167, 0.6667, 0.001, 93, 86.35, 1.098, 0.038,
0.325, 89, 3, 133, 35.82, 0.222, 1000, 98000)`.

`single_literals = true` rounds each through Float32 first, which is what
Fortran does to a bare literal (default REAL) before widening it to meet a
double operand. That reading reproduces CUTEst's objective at x0 to all 16
digits and the Float64 reading does not -- see the header.
"""
function hs67_consts(::Type{S}; single_literals::Bool = true) where {S}
    cv = c -> single_literals ? S(Float32(c)) : S(c)
    return (cv(1.6), cv(1.22), cv(0.01), cv(112.0), cv(13.167), cv(0.6667),
            cv(0.001), cv(93.0), cv(86.35), cv(1.098), cv(0.038), cv(0.325),
            cv(89.0), cv(3.0), cv(133.0), cv(35.82), cv(0.222), cv(1000.0),
            cv(98000.0))
end

"""
    hs67_elements(x::NTuple{3,T}, k::NTuple{19,S}; maxit = 64) -> NTuple{8,T}

Every element value HS67 needs, from the three problem variables: slots
1..7 are `Y2, Y3, Y4, Y5, Y6, Y7, Y8` (the SIF element types of the same
names) and slot 8 is `Y2 * Y5` (element type `Y2Y5`).

This is the whole of the Fortran's `HS67`, value part. `k` is
[`hs67_consts`](@ref), kept in the working precision while `x` may be a
`ForwardDiff.Dual` over it. Both fixed-point loops run exactly as the Fortran
runs them: same first guess, same update, same `> 0.001` absolute stopping
rule, and the returned `Y2`/`Y4` are the iterates BEFORE the step that failed
the test. Differentiating this function therefore differentiates the iterates,
which is what the Fortran's hand-coded `G`/`H` recursions do.

`maxit` bounds each loop so the function cannot spin; x0 needs 5 and 12.
"""
@noinline function hs67_elements(x::NTuple{3,T}, k::NTuple{19,S};
                                 maxit::Int = 64) where {T,S}
    x1, x2, x3 = x
    c16, c122, c001, c112, c13167, c06667, ctol, c93, c8635, c1098, c0038,
        c0325, c89, c3, c133, c3582, c0222, c1000, c98000 = k

    # --- Y2 by fixed-point iteration; Y3 and Y6 fall out of the final pass
    y2 = c16 * x1
    y3 = zero(y2)
    y6 = zero(y2)
    for _ in 1:maxit
        y3 = c122 * y2 - x1
        y6 = (x2 + y3) / x1
        y2c = c001 * x1 * (c112 + c13167 * y6 - c06667 * y6 * y6)
        abs(y2c - y2) > ctol || break
        y2 = y2c
    end

    # --- Y4 by fixed-point iteration; Y5, Y8 and Y7 fall out of the final pass
    y4 = T(c93)
    y5 = zero(y2)
    y7 = zero(y2)
    y8 = zero(y2)
    for _ in 1:maxit
        y5 = c8635 + c1098 * y6 - c0038 * y6 * y6 + c0325 * (y4 - c89)
        y8 = c3 * y5 - c133
        y7 = c3582 - c0222 * y8
        y4c = c98000 * x3 / (y2 * y7 + c1000 * x3)
        abs(y4c - y4) > ctol || break
        y4 = y4c
    end

    return (y2, y3, y4, y5, y6, y7, y8, y2 * y5)
end

"""
    hs67_elements(x::NTuple{3,T}; single_literals = true, maxit = 64)

Convenience method that builds the constants in `T` itself.
"""
hs67_elements(x::NTuple{3,T}; single_literals::Bool = true, maxit::Int = 64) where {T} =
    hs67_elements(x, hs67_consts(T; single_literals); maxit)

"HS67 element type name -> slot in [`hs67_elements`](@ref)'s output."
const HS67_SLOTS = Dict("Y2" => 1, "Y3" => 2, "Y4" => 3, "Y5" => 4,
                        "Y6" => 5, "Y7" => 6, "Y8" => 7, "Y2Y5" => 8)

"The three elemental variables of every HS67 element type, in SIF order."
const HS67_EVARS = (:U1, :U2, :U3)

# ---------------------------------------------------------------------------
# HS85: the value cascade

"""
    hs85_elements(x::NTuple{5,T}) -> NTuple{39,T}

Every element value HS85 needs, from the five problem variables: slots 1..20
are `Y(1..20)` and slots 21..39 are `C(1..19)`, matching `YF(I)` and `CF(I)`
in the SIF's `INDIVIDUALS`.

This is the whole of `IFUN85`, value part: the Hock-Schittkowski 85 wood-pulp
plant cascade, in the Fortran's own order and with its own grouping of
operations, so Float64 evaluation reproduces the reference's bit pattern and
not merely its value. All literals are double in the source, so they are
written plainly here; `T(...)` keeps a Float32 model in Float32 and a BigFloat
model's constants equal to the Fortran's.

Straight-line, allocation-free, branch-free.
"""
@noinline function hs85_elements(x::NTuple{5,T}) where {T}
    x1, x2, x3, x4, x5 = x

    y1 = x2 + x3 + T(41.6)
    c1 = T(0.024) * x4 - T(4.62)
    y2 = T(12.0) + T(12.5) / c1
    c2 = T(3.535e-4) * x1 * x1 + T(0.5311) * x1 + T(0.08705) * x1 * y2
    c3 = T(0.052) * x1 + T(78.0) + T(2.377e-3) * x1 * y2
    y3 = c2 / c3
    y4 = T(19.0) * y3
    c4 = T(0.04782) * (x1 - y3) + T(0.1956) * ((x1 - y3) * (x1 - y3)) / x2 +
         T(0.6376) * y4 + T(1.594) * y3
    c5 = T(100.0) * x2
    c6 = x1 - y3 - y4
    c7 = T(0.95) - c4 / c5
    y5 = c6 * c7
    y6 = x1 - y3 - y4 - y5
    c8 = T(0.995) * (y4 + y5)
    y7 = c8 / y1
    y8 = c8 / T(3798.0)
    c9 = y7 - T(0.0663) * y7 / y8 - T(0.3153)
    y9 = T(96.82) / c9 + T(0.321) * y1
    y10 = T(2.29) * y3 + T(1.258) * y4 + T(1.29) * y5 + T(1.71) * y6
    y11 = T(1.71) * x1 + T(0.58) * y3 - T(0.452) * y4
    c10 = T(12.3) / T(752.3)
    c11 = T(1.74125) * x1 * y2
    c12 = T(0.9995) * y10 + T(1998.0)
    y12 = c10 * x1 + c11 / c12
    y13 = c12 - T(1.75) * y2
    y14 = T(3623.0) + T(64.4) * x2 + T(58.4) * x3 + T(1.46312e5) / (y9 + x5)
    c13 = T(0.9995) * y10 - T(0.1121) * y14 + T(60.8) * x2 + T(48.0) * x4 -
          T(5095.0)
    y15 = y13 / c13
    y16 = T(1.48e5) + T(-3.31e5) * y15 + T(40.0) * y13 + T(-6.1e1) * y15 * y13
    c14 = T(2324.0) * y10 - T(2.874e7) * y2
    y17 = T(1.413e7) + T(-1.328e3) * y10 + T(-5.31e2) * y11 + c14 / c12
    c15 = y13 / y15 - y13 / T(0.52)
    c16 = T(1.104) - T(0.72) * y15
    c17 = y9 + x5
    c18 = c15 / c16
    c19 = y2 / c12
    y18 = y4 + (T(-2.8e-1) / T(7.2e-1)) / y5
    y19 = T(-3.496e3) * c19
    y20 = T(6.2212e4) / c17 - y1

    return (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
            y11, y12, y13, y14, y15, y16, y17, y18, y19, y20,
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10,
            c11, c12, c13, c14, c15, c16, c17, c18, c19)
end

"The five elemental variables of HS85's `Y`/`C` element types, in SIF order."
const HS85_EVARS = (:X1, :X2, :X3, :X4, :X5)

# ---------------------------------------------------------------------------
# AD over the value functions: one gradient sweep, one Hessian of a weighted
# sum. `vals` is any `NTuple{N,T} -> NTuple{NS,T}`.

@inline _seed(::Type{S}, ::Val{N}, i::Int) where {S,N} =
    ForwardDiff.Partials(ntuple(j -> ifelse(i == j, one(S), zero(S)), Val(N)))

"""
    _slot_jac(vals, v::NTuple{N,T}) -> NTuple{NS,Partials{N,T}}

The full Jacobian of a slot-value function, in one forward sweep: entry `s`
holds the gradient of slot `s`.
"""
@inline function _slot_jac(vals, v::NTuple{N,T}) where {N,T}
    D = ForwardDiff.Dual{:HSJ,T,N}
    vd = ntuple(i -> D(v[i], _seed(T, Val(N), i)), Val(N))
    return _slot_partials(vals(vd))
end

# `ntuple` over the slot count, not `map`: Base's `map` on a tuple recurses
# with a splat, which for HS85's 39 slots exceeds what inference unrolls and
# falls back to a `Vector{Any}`. That is merely slow on the host, but a heap
# allocation inside a kernel is an `InvalidIRError`.
@inline _slot_partials(out::NTuple{NS,Any}) where {NS} =
    ntuple(s -> ForwardDiff.partials(out[s]), Val(NS))

"""
    _slot_hess(vals, v::NTuple{N,T}, w::NTuple{NS,T}) -> NTuple{N,Partials{N,T}}

The Hessian of `sum_s w[s] * vals(v)[s]`, by twice-nested forward mode. Row
`i` is returned as a `Partials{N,T}`; the matrix is symmetric, so only its
lower triangle is ever read.
"""
@inline function _slot_hess(vals, v::NTuple{N,T}, w::NTuple{NS,T}) where {N,NS,T}
    D1 = ForwardDiff.Dual{:HSH1,T,N}
    D2 = ForwardDiff.Dual{:HSH2,D1,N}
    vd = ntuple(i -> D2(D1(v[i], _seed(T, Val(N), i)), _seed(D1, Val(N), i)),
                Val(N))
    out = vals(vd)
    s = w[1] * out[1]
    for j in 2:NS
        s += w[j] * out[j]
    end
    p = ForwardDiff.partials(s)
    return ntuple(i -> ForwardDiff.partials(p[i]), Val(N))
end

# ---------------------------------------------------------------------------
# Assembly

"""
    HSRow

One SIF group of an HS67/HS85 model after decoding: its linear terms (variable
index, coefficient), the element slots it reads (slot, weight) and the group
constant, which the row VALUE carries with a minus sign (CUTEst's convention,
matching `ExaModelsSIF.build_exa`).
"""
struct HSRow
    lin::Vector{Tuple{Int,Float64}}
    slots::Vector{Tuple{Int,Float64}}
    constant::Float64
end

"""
    _hs_rows(P, slotof) -> (objrow, conrows, connames, idx)

Split a `sif_decode(...; external = :defer)` problem into its objective group
and its constraint rows, resolving each element to a slot of the value
function through `slotof(element) -> Int`.

`idx` is the variable index of each elemental variable, shared by every
element -- these problems map the element's variables to the problem's
variables identically everywhere, and a disagreement is an error rather than
something to paper over. Group scales must be 1 (they are) because nothing
here divides by them.
"""
function _hs_rows(P, slotof, evars::NTuple{N,Symbol}) where {N}
    idx = nothing
    mk = function (gn)
        g = P.groups[gn]
        g.scale == 1.0 || error("$gn: group scale $(g.scale) is not supported here")
        lin = Tuple{Int,Float64}[(j, a) for (j, a) in g.lin]
        slots = Tuple{Int,Float64}[]
        for (en, w) in g.elems
            e = P.elements[en]
            ix = ntuple(a -> e.varmap[evars[a]], Val(N))
            if idx === nothing
                idx = ix
            else
                idx == ix || error("element $en maps its variables to $ix, " *
                                   "but an earlier element used $idx")
            end
            push!(slots, (slotof(e), w))
        end
        return HSRow(lin, slots, g.constant)
    end

    objname = nothing
    connames = String[]
    for gn in P.grouporder
        if P.groups[gn].kind == 'N'
            objname === nothing || error("more than one objective group")
            objname = gn
        else
            push!(connames, gn)
        end
    end
    objname === nothing && error("no objective group")
    objrow = mk(objname)
    conrows = [mk(gn) for gn in connames]
    idx === nothing && error("no external elements found")
    return objrow, conrows, connames, idx
end

"Row bounds for a decoded group, with the constant folded into the value."
function _hs_bounds(g, ::Type{T}) where {T}
    r = g.range_
    if g.kind == 'E'
        r === nothing && return (zero(T), zero(T))
        return r >= 0 ? (zero(T), T(r)) : (T(r), zero(T))
    elseif g.kind == 'G'
        return (zero(T), r === nothing ? T(Inf) : T(abs(r)))
    elseif g.kind == 'L'
        return (r === nothing ? T(-Inf) : T(-abs(r)), zero(T))
    end
    error("unexpected constraint kind $(g.kind)")
end

"""
    _hs_oracle_model(P, vals, slotof, evars, nslot; T) -> ExaModel

Build an HS67/HS85-shaped model: one `ScalarNonlinearOracle` objective and one
`VectorNonlinearOracle` holding every constraint row in SIF group order.

`vals` is the generic value function (`NTuple{N,T} -> NTuple{nslot,T}`),
`slotof` resolves a decoded element to one of its slots, and `evars` names the
element type's variables. Derivatives come from ForwardDiff over `vals`; see
[`_slot_jac`](@ref) and [`_slot_hess`](@ref).
"""
function _hs_oracle_model(P, vals, slotof, evars::NTuple{N,Symbol},
                          ::Val{NS}; T = Float64) where {N,NS}
    all(==(1.0), P.varscale) || error("variable scaling is not supported here")
    objrow, conrows, connames, idx = _hs_rows(P, slotof, evars)
    nvar = length(P.varnames)
    ncon = length(conrows)

    lcon = zeros(T, ncon)
    ucon = zeros(T, ncon)
    for (i, gn) in enumerate(connames)
        lo, hi = _hs_bounds(P.groups[gn], T)
        lcon[i] = lo <= -1e20 ? T(-Inf) : lo
        ucon[i] = hi >= 1e20 ? T(Inf) : hi
    end

    xt = (xv, E) -> ntuple(a -> E(xv[idx[a]]), Val(N))

    # --- objective: linear terms + weighted slots - constant, all in one
    # ScalarNonlinearOracle (ExaModels does not promise an expression
    # objective summed with an oracle one).
    objlin = objrow.lin
    objslots = objrow.slots
    objconst = objrow.constant
    objw = (E) -> ntuple(s -> begin
                             acc = zero(E)
                             for (sl, w) in objslots
                                 sl == s && (acc += E(w))
                             end
                             acc
                         end, Val(NS))

    fobj = function (xv)
        E = eltype(xv)
        out = vals(xt(xv, E))
        acc = -E(objconst)
        for (j, a) in objlin
            acc += E(a) * xv[j]
        end
        for (sl, w) in objslots
            acc += E(w) * out[sl]
        end
        return acc
    end

    gobj! = function (g, xv)
        E = eltype(g)
        fill!(g, zero(E))
        for (j, a) in objlin
            g[j] += E(a)
        end
        J = _slot_jac(vals, xt(xv, E))
        for (sl, w) in objslots
            p = J[sl]
            for a in 1:N
                g[idx[a]] += E(w) * p[a]
            end
        end
        return g
    end

    hvpobj! = function (Hv, xv, v)
        E = eltype(Hv)
        fill!(Hv, zero(E))
        H = _slot_hess(vals, xt(xv, E), objw(E))
        for a in 1:N
            acc = zero(E)
            for b in 1:N
                acc += H[a][b] * E(v[idx[b]])
            end
            Hv[idx[a]] += acc
        end
        return Hv
    end

    # one lower triangle of the n x n block the elements share
    hr = Int[]
    hc = Int[]
    for a in 1:N, b in 1:a
        push!(hr, max(idx[a], idx[b]))
        push!(hc, min(idx[a], idx[b]))
    end

    c = ExaModels.ExaCore(T)
    c, _ = ExaModels.add_var(c, nvar; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    sor = ExaModels.ScalarNonlinearOracle(;
        nvar = nvar, f = fobj, grad! = gobj!, hvp! = hvpobj!,
        hess_rows = copy(hr), hess_cols = copy(hc), adapt = Val(true))
    c = ExaModels.objective(c, sor)

    if ncon > 0
        # Jacobian sparsity: a row's linear terms, then a dense block of N for
        # every element it holds. Coordinate duplicates are summed by
        # ExaModels, so an overlap between the two would be harmless.
        jr = Int[]
        jc = Int[]
        for (i, row) in enumerate(conrows)
            for (j, _) in row.lin
                push!(jr, i); push!(jc, j)
            end
            for _ in row.slots, a in 1:N
                push!(jr, i); push!(jc, idx[a])
            end
        end

        f! = function (cv, xv)
            E = eltype(cv)
            out = vals(xt(xv, E))
            for (i, row) in enumerate(conrows)
                acc = -E(row.constant)
                for (j, a) in row.lin
                    acc += E(a) * xv[j]
                end
                for (sl, w) in row.slots
                    acc += E(w) * out[sl]
                end
                cv[i] = acc
            end
            return cv
        end

        jac! = function (jv, xv)
            E = eltype(jv)
            J = _slot_jac(vals, xt(xv, E))
            p = 0
            for row in conrows
                for (_, a) in row.lin
                    jv[p += 1] = E(a)
                end
                for (sl, w) in row.slots
                    g = J[sl]
                    for a in 1:N
                        jv[p += 1] = E(w) * g[a]
                    end
                end
            end
            return jv
        end

        # Every row reads the same variables, so sum the multipliers into one
        # weight per slot and take a single Hessian of that weighted sum.
        hess! = function (hv, xv, y)
            E = eltype(hv)
            w = ntuple(s -> begin
                           acc = zero(E)
                           for (i, row) in enumerate(conrows)
                               for (sl, ww) in row.slots
                                   sl == s && (acc += E(ww) * E(y[i]))
                               end
                           end
                           acc
                       end, Val(NS))
            H = _slot_hess(vals, xt(xv, E), w)
            p = 0
            for a in 1:N, b in 1:a
                hv[p += 1] = H[a][b]
            end
            return hv
        end

        vor = ExaModels.VectorNonlinearOracle(;
            nvar = nvar, ncon = ncon,
            jac_rows = jr, jac_cols = jc, hess_rows = hr, hess_cols = hc,
            lcon = lcon, ucon = ucon,
            f! = f!, jac! = jac!, hess! = hess!, adapt = Val(true))
        c = ExaModels.constraint(c, vor)
    end
    return ExaModels.ExaModel(c)
end

"""
    hs67_model(; T = Float64, backend = nothing, single_literals = true,
               maxit = 64) -> ExaModel

HS67 (Colville's refinery / Hock-Schittkowski 67): minimize
`5.04 x1 + 0.035 x2 + 10 x3 - 0.063 Y2 Y5 + 3.36 Y3` subject to the fourteen
one-sided bounds on `Y2..Y8`, where the `Y` are the external Fortran's
intermediate quantities -- two of them defined by a fixed-point iteration that
this port reproduces step for step. See [`hs67_elements`](@ref) and the file
header; `single_literals = false` switches to the (wrong, measured) Float64
reading of the Fortran's bare constants.
"""
function hs67_model(; T = Float64, backend = nothing,
                    single_literals::Bool = true, maxit::Int = 64)
    backend === nothing || error(
        "hs67_model does not take a backend: its oracle callbacks are host " *
        "loops. Use hs67_kernel_model(; backend = ...), which evaluates the " *
        "same hs67_elements from KernelAbstractions kernels.")
    P = ExaModelsSIF.sif_decode(sif_problem_path("HS67"); external = :defer)
    _hs67_check_types(P)
    k = hs67_consts(T; single_literals)
    vals = v -> hs67_elements(v, k; maxit)
    return _hs_oracle_model(P, vals, _hs67_slotof, HS67_EVARS, Val(8); T)
end

"Check that HS67's decode defers exactly the element types the port covers."
function _hs67_check_types(P)
    for (tname, fn) in P.external_types
        (haskey(HS67_SLOTS, tname) && fn === :HS67) || error(
            "HS67: expected deferred element types $(sort(collect(keys(HS67_SLOTS)))) " *
            "calling HS67, found $tname => $fn")
    end
    return nothing
end

"HS67 decoded element -> slot of [`hs67_elements`](@ref)'s output."
_hs67_slotof(e) = get(HS67_SLOTS, e.etype) do
    error("HS67: element of unexpected type $(e.etype)")
end

"""
    hs85_model(; T = Float64, backend = nothing) -> ExaModel

HS85 (Hock-Schittkowski 85): maximize the net profit of a hypothetical
wood-pulp plant in five variables, with twenty-one rows -- one linear
(`1.5 x2 - x3 >= 0`) and twenty two-sided or one-sided bounds on the external
Fortran's `Y(1..20)`. The objective reads `Y5, Y12, Y13, Y14, Y16, Y17` and
`C18, C19` from the same cascade. See [`hs85_elements`](@ref).
"""
function hs85_model(; T = Float64, backend = nothing)
    backend === nothing || error(
        "hs85_model does not take a backend: its oracle callbacks are host " *
        "loops. Use hs85_kernel_model(; backend = ...), which evaluates the " *
        "same hs85_elements from KernelAbstractions kernels.")
    P = ExaModelsSIF.sif_decode(sif_problem_path("HS85"); external = :defer)
    _hs85_check_types(P)
    vals = v -> hs85_elements(v)
    return _hs_oracle_model(P, vals, _hs85_slotof, HS85_EVARS, Val(39); T)
end

"Check that HS85's decode defers exactly the element types the port covers."
function _hs85_check_types(P)
    for (tname, fn) in P.external_types
        (tname in ("Y", "C") && fn === :IFUN85) || error(
            "HS85: expected deferred Y/C element types calling IFUN85, found " *
            "$tname => $fn")
    end
    return nothing
end

"HS85 decoded element -> slot of [`hs85_elements`](@ref)'s output."
function _hs85_slotof(e)
    i = Int(floor(e.parvals[:PI]))               # the Fortran's INT( PI )
    1 <= i <= 20 || error("HS85: element index $i out of 1:20")
    if e.etype == "Y"
        return i
    elseif e.etype == "C"
        i <= 19 || error("HS85: C element index $i out of 1:19")
        return 20 + i
    end
    error("HS85: element of unexpected type $(e.etype)")
end

# ---------------------------------------------------------------------------
# The same two value cascades called from KernelAbstractions kernels, with the
# row tables resident on the model's backend.
#
# WHAT IS PARALLEL: nothing, and nothing can be. HS67's cascade contains a
# fixed-point iteration and HS85's is one long dependency chain; every row of
# either problem reads the SAME cascade. So each kernel runs with
# `ndrange = 1`: one thread evaluates the cascade once and then writes all 14
# (HS67) or 21 (HS85) rows from it. Evaluating per row instead would recompute
# the cascade 14 or 21 times for no gain.
#
# The reason to have a kernel at all is that these rows must be evaluable with
# their data on a device and scalar indexing disabled
# (`CUDA.allowscalar(false)`), which a host loop over a device array cannot be.
# It is not a throughput claim; 3 and 5 variables will never want a GPU.
#
# ONE BUILDER FOR BOTH BACKENDS. `backend = nothing` runs the same kernels on
# KernelAbstractions' CPU backend, so a CPU-versus-GPU comparison tests the
# device rather than two unrelated implementations; `test/hs_gpu_verify.jl` is
# that comparison, and it also compares the host kernels against
# `_hs_oracle_model`, which is what was verified against the Fortran.
#
# THE ROW TABLES ARE DENSE, AND THE JACOBIAN WITH THEM. `A[i, j]` holds row
# i's linear coefficient for variable j and `W[i, s]` its weight on slot s, so
# a kernel needs no per-row offsets into a packed list. Both problems have
# N == nvar, and `_hs_oracle_model` already declares a dense N-block for every
# row that holds an element, so the two Jacobians differ only on a row that
# holds none: HS85's purely linear row gains three structural zeros here,
# taking nnzj from 102 to 105. At 14 x 3 and 21 x 5 nothing is worth packing.
# The orders differ too -- the host interleaves each row's linear entries with
# its element blocks -- so the verification compares the two assembled rather
# than coordinate by coordinate.
#
# No `@inbounds`: the gathers go through `idx`, checked once at build time by
# `_hs_check_indices`, and the oracle value buffers are written at the lengths
# ExaModels declared for them. ExaModels' oracle contract hands each callback a
# dense one-based buffer of that length.

"""
    _hs_check_indices(idx, nvar)

Check that the elemental-variable indices are a permutation of `1:nvar`.

Two kernel properties rest on this and on nothing else. The gathers
`x[idx[a]]` have no bounds check of their own, and on a device an out-of-range
gather has no useful diagnosis -- it reads whatever memory is there. And the
`hvp` kernels ASSIGN rather than accumulate into `Hv[idx[a]]`, which fills the
whole buffer only because `idx` is onto `1:nvar`. The check runs once per
build.
"""
function _hs_check_indices(idx::NTuple{N,Int}, nvar::Integer) where {N}
    N == nvar || error("the element has $N elemental variables but the " *
                       "problem has $nvar")
    seen = falses(nvar)
    for (a, j) in pairs(idx)
        1 <= j <= nvar ||
            error("elemental variable $a maps to index $j, outside 1:$nvar")
        seen[j] && error("elemental variable $a repeats variable index $j")
        seen[j] = true
    end
    return nothing
end

@inline _hs_gather(x, idx::NTuple{N,<:Integer}) where {N} =
    ntuple(a -> x[idx[a]], Val(N))

# Slot reductions, as plain loops over the slot index with the count taken
# from the value tuple's type. NOT an unrolled tuple reduction: HS85 has 39
# slots, and a 39-deep recursion over `Base.tail` is past the length inference
# unrolls, so it falls back to a boxed accumulator -- merely slow on the host,
# but a heap allocation inside a kernel is an `InvalidIRError`.
@inline function _hs_slotval(W, i, out::NTuple{NS,T}) where {NS,T}
    acc = zero(T)
    for s in 1:NS
        acc += T(W[i, s]) * out[s]
    end
    return acc
end

@inline function _hs_slotgrad(W, i, J::NTuple{NS,ForwardDiff.Partials{N,T}},
                              a) where {NS,N,T}
    acc = zero(T)
    for s in 1:NS
        acc += T(W[i, s]) * J[s][a]
    end
    return acc
end

"""
    _hs_slotmults(W, y, Val(NS)) -> NTuple{NS}

`sum_i y[i] * W[i, s]`, one weight per slot: the multipliers collapsed into a
single weighted sum of element values, whose Hessian is then one AD call
instead of one per row.
"""
@inline _hs_slotmults(W, y, ::Val{NS}) where {NS} =
    ntuple(s -> begin
               acc = zero(eltype(W))
               for i in axes(W, 1)
                   acc += W[i, s] * y[i]
               end
               acc
           end, Val(NS))

# --- kernels. `idx` is passed by value as a tuple, not as an array: N <= 5, so
# the indices live in registers and `Val(N)` comes out of the tuple's type. The
# objective is the one-row case -- its `A` and `W` have a single row and its
# `y` is one 1.0 -- so it shares the value, Hessian and Hessian-vector kernels
# with the constraint block.

@kernel function _hs_val_kernel!(cv, vals, @Const(x), idx, @Const(A), @Const(W),
                                 @Const(cst))
    @index(Global)
    out = vals(_hs_gather(x, idx))
    for i in eachindex(cv)
        acc = -cst[i]
        for j in axes(A, 2)
            acc += A[i, j] * x[j]
        end
        cv[i] = acc + _hs_slotval(W, i, out)
    end
end

@kernel function _hs_grad_kernel!(g, vals, @Const(x), idx, @Const(A), @Const(W))
    @index(Global)
    J = _slot_jac(vals, _hs_gather(x, idx))
    for a in eachindex(idx)
        j = idx[a]
        g[j] = A[1, j] + _hs_slotgrad(W, 1, J, a)
    end
end

# Dense by rows: slot (i - 1) * N + a holds row i's derivative in variable
# idx[a], which is the order `_hs_kernel_model` declares.
@kernel function _hs_jac_kernel!(jv, vals, @Const(x), idx, @Const(A), @Const(W))
    @index(Global)
    J = _slot_jac(vals, _hs_gather(x, idx))
    n = length(idx)
    for i in axes(W, 1)
        for a in eachindex(idx)
            jv[(i - 1) * n + a] = A[i, idx[a]] + _hs_slotgrad(W, i, J, a)
        end
    end
end

@kernel function _hs_hess_kernel!(hv, vals, @Const(x), idx, @Const(W), @Const(y),
                                  ns::Val)
    @index(Global)
    H = _slot_hess(vals, _hs_gather(x, idx), _hs_slotmults(W, y, ns))
    q = 0
    for a in eachindex(H), b in 1:a
        q += 1
        hv[q] = H[a][b]
    end
end

# One thread owns every entry of `Hv` it writes, because `idx` is a permutation
# of 1:nvar, so this assigns and needs no atomic and no preceding fill.
@kernel function _hs_hvp_kernel!(Hv, vals, @Const(x), idx, @Const(W), @Const(y),
                                 @Const(vv), ns::Val)
    @index(Global)
    H = _slot_hess(vals, _hs_gather(x, idx), _hs_slotmults(W, y, ns))
    vl = _hs_gather(vv, idx)
    for a in eachindex(H)
        acc = zero(eltype(Hv))
        for b in eachindex(vl)
            acc += H[a][b] * vl[b]
        end
        Hv[idx[a]] = acc
    end
end

"""
    _HS67Vals{S}(k, maxit)

HS67's value cascade as a callable value: `v -> hs67_elements(v, k; maxit)`,
with the constants and the iteration bound as fields.

A kernel argument has to be a bits type, which a closure over a keyword call
is not guaranteed to be; this is.
"""
struct _HS67Vals{S}
    k::NTuple{19,S}
    maxit::Int
end
@inline (f::_HS67Vals)(v) = hs67_elements(v, f.k; maxit = f.maxit)

"HS85's value cascade as a callable value; see [`_HS67Vals`](@ref)."
struct _HS85Vals end
@inline (::_HS85Vals)(v) = hs85_elements(v)

"""
    _hs_row_tables(objrow, conrows, nvar, ::Val{NS}, ::Type{T})
        -> (oA, oW, oC, cA, cW, cC)

The decoded rows as dense tables: `A[i, j]` is row i's linear coefficient for
variable j, `W[i, s]` its weight on element slot s, and `C[i]` its group
constant, which the row value carries with a minus sign. The objective is the
one-row block `(oA, oW, oC)`.
"""
function _hs_row_tables(objrow::HSRow, conrows::Vector{HSRow}, nvar::Int,
                        ::Val{NS}, ::Type{T}) where {NS,T}
    fill_rows = function (rows)
        A = zeros(T, length(rows), nvar)
        W = zeros(T, length(rows), NS)
        C = zeros(T, length(rows))
        for (i, row) in pairs(rows)
            for (j, a) in row.lin
                A[i, j] += T(a)
            end
            for (sl, w) in row.slots
                W[i, sl] += T(w)
            end
            C[i] = T(row.constant)
        end
        return A, W, C
    end
    oA, oW, oC = fill_rows([objrow])
    cA, cW, cC = fill_rows(conrows)
    return oA, oW, oC, cA, cW, cC
end

"""
    _hs_kernel_model(P, vals, slotof, evars, Val(NS); T, backend) -> ExaModel

Build an HS67/HS85-shaped model whose callbacks are KernelAbstractions kernels
over `vals`, with the row tables resident on `backend`.

Same shape as [`_hs_oracle_model`](@ref) -- one `ScalarNonlinearOracle`
objective and one `VectorNonlinearOracle` holding every constraint row in SIF
group order -- and the same derivatives, by [`_slot_jac`](@ref) and
[`_slot_hess`](@ref) over the same `vals`. It differs in declaring a dense
Jacobian and in supplying an explicit `hess!` for the objective, which is what
lets the objective's Hessian be assembled in one call on a device instead of
one `hvp!` per column.
"""
function _hs_kernel_model(P, vals, slotof, evars::NTuple{N,Symbol},
                          ::Val{NS}; T = Float64, backend = nothing) where {N,NS}
    all(==(1.0), P.varscale) || error("variable scaling is not supported here")
    objrow, conrows, connames, idx = _hs_rows(P, slotof, evars)
    nvar = length(P.varnames)
    ncon = length(conrows)
    _hs_check_indices(idx, nvar)

    lcon = zeros(T, ncon)
    ucon = zeros(T, ncon)
    for (i, gn) in pairs(connames)
        lo, hi = _hs_bounds(P.groups[gn], T)
        lcon[i] = lo <= -1e20 ? T(-Inf) : lo
        ucon[i] = hi >= 1e20 ? T(Inf) : hi
    end

    oAh, oWh, oCh, cAh, cWh, cCh =
        _hs_row_tables(objrow, conrows, nvar, Val(NS), T)
    cv = v -> ExaModels.convert_array(v, backend)
    oA, oW, oC = cv(oAh), cv(oWh), cv(oCh)
    cA, cW, cC = cv(cAh), cv(cWh), cv(cCh)
    # The objective carries no multiplier; the shared kernels take one anyway.
    one1 = cv(ones(T, 1))
    objbuf = cv(zeros(T, 1))
    ka = backend === nothing ? KernelAbstractions.CPU() : backend
    ns = Val(NS)
    sync = () -> KernelAbstractions.synchronize(ka)

    # One lower triangle of the nvar x nvar block every element shares, in the
    # order `_hs_hess_kernel!` writes it.
    hr = Int[]
    hc = Int[]
    for a in 1:N, b in 1:a
        push!(hr, max(idx[a], idx[b]))
        push!(hc, min(idx[a], idx[b]))
    end

    c = ExaModels.ExaCore(T; backend = backend)
    c, _ = ExaModels.add_var(c, nvar; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    sor = ExaModels.ScalarNonlinearOracle(;
        nvar = nvar,
        f = xv -> begin
            _hs_val_kernel!(ka)(objbuf, vals, xv, idx, oA, oW, oC; ndrange = 1)
            sync()
            return sum(objbuf)
        end,
        grad! = (g, xv) -> begin
            _hs_grad_kernel!(ka)(g, vals, xv, idx, oA, oW; ndrange = 1)
            sync()
            return g
        end,
        hess! = (hv, xv) -> begin
            _hs_hess_kernel!(ka)(hv, vals, xv, idx, oW, one1, ns; ndrange = 1)
            sync()
            return hv
        end,
        hvp! = (Hv, xv, vv) -> begin
            _hs_hvp_kernel!(ka)(Hv, vals, xv, idx, oW, one1, vv, ns; ndrange = 1)
            sync()
            return Hv
        end,
        hess_rows = copy(hr), hess_cols = copy(hc), adapt = Val(false))
    c = ExaModels.objective(c, sor)

    if ncon > 0
        jr = Int[]
        jc = Int[]
        for i in 1:ncon, a in 1:N
            push!(jr, i)
            push!(jc, idx[a])
        end
        vor = ExaModels.VectorNonlinearOracle(;
            nvar = nvar, ncon = ncon,
            jac_rows = jr, jac_cols = jc, hess_rows = hr, hess_cols = hc,
            lcon = lcon, ucon = ucon,
            f! = (cvv, xv) -> begin
                _hs_val_kernel!(ka)(cvv, vals, xv, idx, cA, cW, cC; ndrange = 1)
                sync()
                return cvv
            end,
            jac! = (jv, xv) -> begin
                _hs_jac_kernel!(ka)(jv, vals, xv, idx, cA, cW; ndrange = 1)
                sync()
                return jv
            end,
            hess! = (hv, xv, y) -> begin
                _hs_hess_kernel!(ka)(hv, vals, xv, idx, cW, y, ns; ndrange = 1)
                sync()
                return hv
            end,
            hvp! = (Hv, xv, y, vv) -> begin
                _hs_hvp_kernel!(ka)(Hv, vals, xv, idx, cW, y, vv, ns; ndrange = 1)
                sync()
                return Hv
            end,
            adapt = Val(false))
        c = ExaModels.constraint(c, vor)
    end
    return ExaModels.ExaModel(c)
end

"""
    hs67_kernel_model(; T = Float64, backend = nothing, single_literals = true,
                      maxit = 64) -> ExaModel

HS67 with its callbacks as KernelAbstractions kernels over
[`hs67_elements`](@ref), the row tables resident on `backend`.

`backend = nothing` builds a host model that runs the same kernels on
KernelAbstractions' CPU backend; `backend = CUDABackend()` builds a device
model that never indexes a device array from the host, so it is usable with
`CUDA.allowscalar(false)`. [`hs67_model`](@ref) builds the same problem through
host loops and is the one verified against the Fortran directly.
"""
function hs67_kernel_model(; T = Float64, backend = nothing,
                           single_literals::Bool = true, maxit::Int = 64)
    P = ExaModelsSIF.sif_decode(sif_problem_path("HS67"); external = :defer)
    _hs67_check_types(P)
    vals = _HS67Vals(hs67_consts(T; single_literals), maxit)
    return _hs_kernel_model(P, vals, _hs67_slotof, HS67_EVARS, Val(8);
                            T, backend)
end

"""
    hs85_kernel_model(; T = Float64, backend = nothing) -> ExaModel

HS85 with its callbacks as KernelAbstractions kernels over
[`hs85_elements`](@ref); see [`hs67_kernel_model`](@ref) for what `backend`
buys and [`hs85_model`](@ref) for the host-loop builder.
"""
function hs85_kernel_model(; T = Float64, backend = nothing)
    P = ExaModelsSIF.sif_decode(sif_problem_path("HS85"); external = :defer)
    _hs85_check_types(P)
    return _hs_kernel_model(P, _HS85Vals(), _hs85_slotof, HS85_EVARS, Val(39);
                            T, backend)
end
