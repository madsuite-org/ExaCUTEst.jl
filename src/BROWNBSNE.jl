# BROWNBSNE: Brown badly scaled problem, nonlinear equations version
# 2 variables, 3 equality constraints
# A(1): x[1] - 1e6 = 0
# B(1): x[2] - 2e-6 = 0
# C(1): x[1]*x[2] - 2 = 0
# All variables free; start: x[i] = 1.0
function brownbsne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; start = T(1.0))
    ExaModels.@add_con(c, x[1] - T(1000000))
    ExaModels.@add_con(c, x[2] - T(0.000002))
    ExaModels.@add_con(c, x[1] * x[2] - T(2))
    return ExaModels.ExaModel(c; kwargs...)
end
