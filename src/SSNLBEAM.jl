# SSNLBEAM: Singly Supported Nonlinear Beam, NI=1000 default
# Source: Maurer & Mittelman (1991); SIF input by Ph. Toint, 1993
# Optimal control of beam energy with trapezoidal discretization
#
# Variables: T(0..NI)=x[1..NI+1], X(0..NI)=x[NI+2..2NI+2], U(0..NI)=x[2NI+3..3NI+3]
# Julia: T(I)=x[I+1], X(I)=x[NI+1+I+1]=x[NI+I+2], U(I)=x[2NI+2+I+1]=x[2NI+I+3]
#
# Objective ENERGY (N type):
#   sum_{I=0}^{NI-1} [H/2*U(I)^2 + H/2*U(I+1)^2 + AH/2*cos(T(I)) + AH/2*cos(T(I+1))]
#
# Constraints (E type):
#   EX(I): X(I+1) - X(I) - H/2*sin(T(I+1)) - H/2*sin(T(I)) = 0
#   ET(I): T(I+1) - T(I) - H/2*U(I+1) - H/2*U(I) = 0
#
# Bounds: T(I) in [-1,1], X(I) in [-0.05, 0.05], U free
#         X(0)=X(NI)=0 fixed
function ssnlbeam_model(NI = 1000; T = Float64, backend = nothing, kwargs...)
    ALPHA = T(350.0)
    H     = T(1) / T(NI)
    H2    = H / T(2)
    AH2   = ALPHA * H / T(2)

    nv = 3 * (NI + 1)
    lvar = fill(T(-Inf), nv)
    uvar = fill(T(Inf), nv)
    start = zeros(T, nv)

    for I in 0:NI
        lvar[I + 1] = T(-1); uvar[I + 1] = T(1)    # T(I)
        lvar[NI + 2 + I] = T(-0.05); uvar[NI + 2 + I] = T(0.05)  # X(I)
        # U(I): free (already Inf/-Inf)
        # Start: 0.05*cos(I*H)
        s = T(0.05) * cos(T(I) * H)
        start[I + 1] = s        # T(I)
        start[NI + 2 + I] = s  # X(I)
    end
    # X(0) = 0 fixed
    lvar[NI + 2] = T(0); uvar[NI + 2] = T(0); start[NI + 2] = T(0)
    # X(NI) = 0 fixed
    lvar[2 * NI + 2] = T(0); uvar[2 * NI + 2] = T(0); start[2 * NI + 2] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Objective: sum_{I=0}^{NI-1} [H/2*(U(I)^2+U(I+1)^2) + AH/2*(cos(T(I))+cos(T(I+1)))]
    # T(I)=x[I+1], U(I)=x[2NI+3+I] (0-indexed)
    # T(I)=x[I+1], U(I)=x[2NI+2+I+1]=x[2*NI+I+3]
    ExaModels.@add_obj(c,
        H2 * (x[2 * NI + d.I + 3]^2 + x[2 * NI + d.I + 4]^2) +
        AH2 * (cos(x[d.I + 1]) + cos(x[d.I + 2]))
        for d in [(I = I,) for I in 0:NI-1])

    # CUTEst order: I-outer (EX(I), ET(I) for each I=0..NI-1)
    for I in 0:NI-1
        # EX(I): X(I+1) - X(I) - H/2*sin(T(I+1)) - H/2*sin(T(I)) = 0
        ExaModels.@add_con(c,
            x[NI + 3 + I] - x[NI + 2 + I] -
            H2 * sin(x[I + 2]) - H2 * sin(x[I + 1]))
        # ET(I): T(I+1) - T(I) - H/2*U(I+1) - H/2*U(I) = 0
        ExaModels.@add_con(c,
            x[I + 2] - x[I + 1] -
            H2 * x[2 * NI + I + 4] - H2 * x[2 * NI + I + 3])
    end

    return ExaModels.ExaModel(c; kwargs...)
end
