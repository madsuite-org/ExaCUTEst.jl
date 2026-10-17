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
# meant to be declared and the name was mistyped.) Which variant reproduces
# CUTEst is decided by measurement, not by reading: see test/vandanium_verify.jl.

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
    vandanium_model(name; T = Float64, backend = nothing, single_xx = true)

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
"""
function vandanium_model(name::AbstractString; T = Float64, backend = nothing,
                         single_xx::Bool = true)
    P = ExaModelsSIF.sif_decode(sif_problem_path(name); external = :defer)
    haskey(P.external_types, "BSPLINE") ||
        error("$name: expected a deferred BSPLINE element type, found " *
              "$(collect(keys(P.external_types)))")
    folded = 0
    for g in values(P.groups)
        isempty(g.elems) && continue
        for (en, w) in g.elems
            e = P.elements[en]
            pv = e.parvals
            bk = vandanium_bspline(pv[:X0], pv[:K], pv[:H], pv[:X]; single_xx)
            push!(g.lin, e.varmap[:A] => w * bk)
            folded += 1
        end
        empty!(g.elems)
    end
    folded > 0 || error("$name: no elements folded; the SIF structure changed")
    # nothing external remains, so the decoder can build it unaided
    delete!(P.external_types, "BSPLINE")
    return ExaModelsSIF.build_exa(P; T, backend)
end
