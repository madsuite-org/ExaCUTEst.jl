# POWELLSQ: Powell singular nonlinear system (2 variables, 2 equality constraints)
# NOR problem: 2 nonlinear equations, no explicit objective (obj = 0)
# G1 (E): x1^2 = 0
# G2 (E): 10*x1/(x1 + 0.1) + 2*x2^2 = 0
# Source: M.J.D. Powell (1970), Buckley#217
# SIF: E-type groups (equality constraints); element SQ = X^2; element A = 10*X/(X+0.1)
# Both variables free (FR DEFAULT), start = [3, 1]
function powellsq_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[3.0, 1.0])
    # G1: x1^2 = 0
    ExaModels.@add_con(c, x[1]^2)
    # G2: 10*x1/(x1+0.1) + 2*x2^2 = 0
    ExaModels.@add_con(c, T(10) * x[1] / (x[1] + T(0.1)) + T(2) * x[2]^2)
    return ExaModels.ExaModel(c; kwargs...)
end
