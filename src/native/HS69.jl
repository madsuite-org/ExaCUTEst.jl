# Salvaged from ExaCUTEst 0.1's hand-written port (git master, pre-0.2) and
# kept because it is a genuine win over the oracle tier: this problem's
# external Fortran computes the standard normal CDF, which Julia computes
# natively and exactly via erf, so the model is a real ExaModel -- validated
# against the Fortran reference and able to run on a GPU.
#
# VERIFIED 2026-10-16, both at x0 and a perturbed point, against a
# CUTEstModel of the same problem: objective, gradient, constraints, Jacobian
# and Lagrangian Hessian all agree to 1e-6, and the CUDA build matches the CPU
# build to 1e-9. (HS67 was the third candidate in the legacy port and was
# REJECTED: it replaces the Fortran's fixed-point iteration with closed-form
# roots, and disagrees with the reference in the 4th digit -- obj -868.6357 vs
# -868.7250. Do not resurrect it without re-deriving the iteration.)
# HS69: Cost-optimal inspection plan (4 variables, 2 equality constraints)
# Source: problem 69 in Hock & Schittkowski (1981)
# SIF classification: OOR2-MN-4-2
#
# Same structure as HS68 but different parameters:
#   A=0.1, B=1000.0, D=1.0, NN=4.0
#   AN = A*NN = 0.4,  DROOTN = D*sqrt(NN) = 2.0
#
# CUTEst variable ordering: x[1]=X2, x[2]=X1, x[3]=X3, x[4]=X4
#   (same as HS68, gradient of obj is zero w.r.t. x[1]=X2)
#
# Objective:
#   AN/X1 - X4*(B*(exp(X1)-1) - X3) / ((exp(X1)-1+X4)*X1)
#
# Constraints (equality, lcon=ucon=0):
#   C1: X3 - 2*PHI(-X2)                               = 0
#   C2: X4 - PHI(DROOTN-X2) - PHI(-DROOTN-X2)         = 0
# where PHI(t) = 0.5 + 0.5*erf(t/sqrt(2)) is the standard normal CDF
#
# Variable bounds:
#   x[1]=X2: [0, 100],  x[2]=X1: [0.0001, 100]
#   x[3]=X3: [0, 2],    x[4]=X4: [0, 2]
# Start: all 1.0
function hs69_model(; T = Float64, backend = nothing, kwargs...)
    A_param  = T(0.1)
    B_param  = T(1000.0)
    NN_param = T(4.0)
    AN       = A_param * NN_param    # = 0.4
    DROOTN   = sqrt(NN_param)        # = 2.0

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # CUTEst ordering: x[1]=X2, x[2]=X1, x[3]=X3, x[4]=X4
    ExaModels.@add_var(c, x, 4;
        lvar  = T[0.0,    0.0001, 0.0, 0.0],
        uvar  = T[100.0, 100.0,   2.0, 2.0],
        start = T[1.0,    1.0,    1.0, 1.0])

    # Objective: AN/X1 - X4*R/(S*X1)
    ExaModels.@add_obj(c, let
        X1 = x[2]; X3 = x[3]; X4 = x[4]
        E = exp(X1)
        R = B_param * (E - T(1)) - X3
        S = E - T(1) + X4
        AN / X1 - X4 * R / (S * X1)
    end)

    # C1: X3 - 2*PHI(0 - X2) = 0  (lcon=ucon=0 by default)
    ExaModels.@add_con(c, let
        X2 = x[1]; X3 = x[3]
        X3 - T(2) * (T(0.5) + T(0.5) * SpecialFunctions.erf((T(0) - X2) / sqrt(T(2))))
    end)

    # C2: X4 - PHI(DROOTN-X2) - PHI(-DROOTN-X2) = 0  (lcon=ucon=0 by default)
    ExaModels.@add_con(c, let
        X2 = x[1]; X4 = x[4]
        sq2      = sqrt(T(2))
        phi_pos  = T(0.5) + T(0.5) * SpecialFunctions.erf((T(DROOTN) - X2) / sq2)
        phi_neg  = T(0.5) + T(0.5) * SpecialFunctions.erf((-T(DROOTN) - X2) / sq2)
        X4 - phi_pos - phi_neg
    end)

    return ExaModels.ExaModel(c; kwargs...)
end
