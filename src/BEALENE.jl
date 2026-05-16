# BEALENE: Beale problem, nonlinear equations version
# 2 variables, 3 equality constraints
# A: x[1]*(1 - x[2]) - 1.5 = 0
# B: x[1]*(1 - x[2]^2) - 2.25 = 0
# C: x[1]*(1 - x[2]^3) - 2.625 = 0
# All variables free; start: x[i] = 1.0
function bealene_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; start = T(1.0))
    ExaModels.@add_con(c, x[1] * (T(1) - x[2]) - T(1.5))
    ExaModels.@add_con(c, x[1] * (T(1) - x[2]^2) - T(2.25))
    ExaModels.@add_con(c, x[1] * (T(1) - x[2]^3) - T(2.625))
    return ExaModels.ExaModel(c; kwargs...)
end
