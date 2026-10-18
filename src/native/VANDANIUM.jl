# VANDANIUMS / VANDANMSLS: a generic Julia port of their external Fortran.
#
# The Fortran computes a cubic B-spline value b(x0, k, h, x). Every argument
# is an element PARAMETER, not a variable, so b is a CONSTANT COEFFICIENT and
# the element -- `A * BK` -- is linear in its single variable A. No AD is
# needed for it at all; the piecewise branching happens on data at build time,
# never on a traced value.
#
# A PRECISION TRAP, and the reason there are two variants. The Fortran has no
# `implicit none`, and `xx` is never declared:
#
#       DOUBLE PRECISION FUNCTION b( x0, k, h, x )
#       DOUBLE PRECISION x0, h, k, x
#       DOUBLE PRECISION xs, twoh
#       xx = x - k * h
#
# Under Fortran's implicit typing a name starting with `x` is default REAL,
# i.e. SINGLE precision, so `xx` is rounded to Float32 while `twoh` and the
# result are double. (`xs` is declared and unused -- it looks like `xx` was
# meant to be declared and the name was mistyped.) THE FIRST DERIVATIVES
# settle which reading is the Fortran, and only they can: with the reference's
# stored precision reproduced (below), `single_xx = true` gives VANDANIUMS'
# Jacobian and VANDANMSLS' gradient bit for bit, while `single_xx = false`
# misses them by 3.4597e-08 and 9.7463e-09 -- four orders of magnitude out and
# nowhere near roundoff, i.e. a different function. The values at x0 cannot
# decide it, because x0 is the origin: cons(x0) is just -constant and obj(x0)
# the sum of those squares, and no B-spline value enters either.
#
# WHAT REMAINS AGAINST THE REFERENCE IS ITS STORED PRECISION, 11 SIGNIFICANT
# DIGITS. SIFdecode writes every double in OUTSDIF.d as `d.dddddddddd D+ee`,
# and each of the 1110 values these rows read is exactly the SIF's value
# rounded to 11 significant digits -- the same mechanism as CAR2/ORBIT2's
# interval endpoints (`src/native/COLLOC.jl`, note 2) and JIMACK's group
# scale, in the same file format. Three arrays carry it here:
#
#     10 group CONSTANTS, YOVERE = (1/E) Y: worst 1.1928422923443212 stored
#        as 1.1928422922999999, 3.7156e-11 relative;
#    220 element WEIGHTS, EINV = 1/E: worst 1.0201042138464864 stored as
#        1.0201042138, 4.5570e-11;
#    880 element PARAMETERS, of which only H = (XU-XL)/N = 5.5/19 is inexact:
#        0.28947368421052633 stored as 2.8947368421D-01, 1.8181e-12. X0, K
#        and X are exact at 11 digits.
#
# Measured against `CUTEstModel` at x0 and at a perturbed point. From the
# SIF's exact values VANDANIUMS' `cons` differs by 2.2418e-11 relative -- 9 of
# its 10 rows not bit-exact at x0, worst F7 at |d| = 4.4836e-11 -- and its
# `jac` by 9.5727e-12 on every one of the 40 nonzero entries, worst (F3, A2);
# VANDANMSLS' obj by 1.9374e-13 and its grad by 6.9954e-12. The quantities
# separate cleanly: the constants carry ALL of `cons` and none of `jac`, while
# `jac` is the weights (9.1815e-12) and H (1.9414e-12) together.
# With all three arrays rounded to 11 digits, every row value and every
# Jacobian entry agrees with the compiled Fortran BIT FOR BIT at both points,
# as does VANDANMSLS' gradient; its objective, which sums the ten squared
# groups in a different order than the Fortran, agrees to 1.2e-16. The digit
# count is measured and not fitted -- 10 digits leaves 2.0e-10 on `cons` and
# 7.3e-11 on `jac`, 12 digits leaves 2.0e-11 and 1.1e-11, and 9, 13 and 14
# fail likewise.
# `ref_param_rounding = true` reproduces it. The default is `false`: the
# full-precision values are what the SIF says, and rounding the decode to
# SIFdecode's output format is a decision for ExaModelsSIF to take for the
# whole collection, not a fudge inside one port.

"""
    vandanium_bspline(x0, k, h, x; single_xx = true)

Value of the k-th cubic B-spline at `x`, as the VANDANIUM* external Fortran
computes it. All arguments are data.

`single_xx = true` rounds the `xx` intermediate through Float32, reproducing
the Fortran's implicit single-precision declaration; `false` keeps it in the
working precision. Generic in the argument type, so it also evaluates in
BigFloat for a precision check.
"""
function vandanium_bspline(x0::T, k::T, h::T, x::T; single_xx::Bool = true) where {T}
    xx_full = x - k * h
    xx = single_xx ? T(Float32(xx_full)) : xx_full
    twoh = h + h
    d = xx - x0
    if xx <= x0 - twoh || xx >= x0 + twoh
        return zero(T)
    elseif xx <= x0 - h
        return (twoh + d^3) / 6
    elseif xx >= x0 + h
        return (twoh - d^3) / 6
    elseif xx <= x0
        return twoh * h * h / 3 - (one(T) / 2) * (twoh + d) * d^2
    else
        return twoh * h * h / 3 - (one(T) / 2) * (twoh - d) * d^2
    end
end

"""
    vandanium_model(name; T = Float64, backend = nothing, prod = :auto,
                    single_xx = true)

Build VANDANIUMS / VANDANMSLS as a genuine ExaModel -- not an oracle.

Because every argument of the B-spline is an element PARAMETER, each element
contributes `w * b(...) * A`: a linear term with a constant coefficient. So
the external Fortran can be CONSTANT-FOLDED into the group's linear part,
after which nothing external remains and the decoder's own machinery builds
the model. That makes these two SIMD-abstracted and `backend`-capable, which
an oracle-backed port would not be.

Contrast with the BA-L* family, where the Fortran depends on the variables and
so must stay a function (and, being ~30 operations deep, must be a callback
rather than an expression). The right shape follows from what the Fortran
actually depends on.

`ref_param_rounding = true` rounds every datum these rows read -- element
parameters, element weights and group constants -- to the 11 significant
decimal digits the reference's `OUTSDIF.d` carries, which makes every row
value and every Jacobian entry agree with the compiled Fortran bit for bit.
The default `false` uses the SIF's exact values; see the header for the
measurement.
"""
function vandanium_model(name::AbstractString; T = Float64, backend = nothing,
                         prod = :auto, single_xx::Bool = true,
                         ref_param_rounding::Bool = false)
    P = ExaModelsSIF.sif_decode(sif_problem_path(name); external = :defer)
    haskey(P.external_types, "BSPLINE") ||
        error("$name: expected a deferred BSPLINE element type, found " *
              "$(collect(keys(P.external_types)))")
    # A datum as the compiled Fortran sees it: the SIF's exact value, or -- with
    # `ref_param_rounding` -- rounded to the 11 significant decimal digits
    # OUTSDIF.d preserves. See the header.
    r(v) = ref_param_rounding ? round(v, sigdigits = 11) : v
    folded = 0
    for g in values(P.groups)
        g.constant = r(g.constant)
        isempty(g.elems) && continue
        for (en, w) in g.elems
            e = P.elements[en]
            pv = e.parvals
            bk = vandanium_bspline(r(pv[:X0]), r(pv[:K]), r(pv[:H]), r(pv[:X]); single_xx)
            push!(g.lin, e.varmap[:A] => r(w) * bk)
            folded += 1
        end
        empty!(g.elems)
    end
    folded > 0 || error("$name: no elements folded; the SIF structure changed")
    # nothing external remains, so the decoder can build it unaided
    delete!(P.external_types, "BSPLINE")
    return ExaModelsSIF.build_exa(P; T, backend, prod)
end
