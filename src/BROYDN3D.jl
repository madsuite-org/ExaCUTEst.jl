# BROYDN3D: Broyden tridiagonal system of nonlinear equations (N variables, N constraints)
# Source: More', Garbow, Hillstrom (1981), problem 30; Toint#17; Buckley#78
# Classification: NQR2-AN-V-V (nonlinear equations, equality constraints, zero objective)
# Default N = 5000, all FREE (FR DEFAULT), start = -1.0
#
# ELEMENT BROY(V1; K1=2): F = (3 - K1*V1)*V1 = (3 - 2*V1)*V1 = 3*V1 - 2*V1^2
# CONSTANTS: Z BROYDN3D E(I) -K2, where -K2 = KAPPA2*(-1.0) = -1.0 for all I
# Group value = linear_part + element_contrib - CONSTANT
#
# E(1):   (3-2*x[1])*x[1] - 2*x[2] - (-1.0) = (3-2*x[1])*x[1] - 2*x[2] + 1
# E(I):   (3-2*x[I])*x[I] - x[I-1] - 2*x[I+1] + 1   for I=2..N-1
# E(N):   (3-2*x[N])*x[N] - x[N-1] + 1
# All E type (equality): lcon = ucon = 0
function broydn3d_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(-1.0))
    # E(1): (3-2*x[1])*x[1] - 2*x[2] + 1 = 0
    ExaModels.@add_con(c, (T(3) - T(2) * x[1]) * x[1] - T(2) * x[2] + T(1))
    # E(I) for I=2..N-1: (3-2*x[I])*x[I] - x[I-1] - 2*x[I+1] + 1 = 0
    ExaModels.@add_con(
        c,
        (T(3) - T(2) * x[i]) * x[i] - x[i-1] - T(2) * x[i+1] + T(1) for i in 2:N-1
    )
    # E(N): (3-2*x[N])*x[N] - x[N-1] + 1 = 0
    ExaModels.@add_con(c, (T(3) - T(2) * x[N]) * x[N] - x[N-1] + T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
