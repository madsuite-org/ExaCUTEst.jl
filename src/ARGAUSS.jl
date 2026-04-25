# ARGAUSS: More's Gaussian problem (3 variables, 15 equality constraints)
# Nonlinear feasibility: g_i(x) = X1*exp(X2*(-0.5*(ti - X3)^2)) - C_i = 0, i=1..15
# where ti = (8-i)*0.5 and C_i are observed constants
# BOUNDS: FR DEFAULT → all variables free
# All groups are XE (equality constraint type); objective is constant 0
# Source: More', Garbow, Hillstrom (1981), problem 9
function argauss_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf, -Inf, -Inf],
                                 uvar = T[Inf, Inf, Inf],
                                 start = T[0.4, 1.0, 0.0])

    # Constants C(I): RHS for each constraint
    ExaModels.@add_par(c, C, T[0.0009, 0.0044, 0.0175, 0.0540, 0.1295, 0.2420,
                                0.3521, 0.3989, 0.3521, 0.2420, 0.1295, 0.0540,
                                0.0175, 0.0044, 0.0009])

    # T parameter: ti = (8-i)*0.5 for i=1..15
    ExaModels.@add_par(c, TI, T[(8 - i) * 0.5 for i in 1:15])

    # GAUSS element: F = V1 * exp(V2 * (-0.5 * (T - V3)^2))
    # Constraint: g_i(x) = F(X1,X2,X3,ti) - C_i = 0
    ExaModels.@add_con(c, x[1] * exp(x[2] * (T(-0.5) * (TI[i] - x[3])^2)) - C[i] for i in 1:15)

    return ExaModels.ExaModel(c; kwargs...)
end
