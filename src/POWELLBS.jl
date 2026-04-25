# POWELLBS: Powell badly scaled problem (More, Garbow & Hillstrom 1981)
# 2 free variables: X1, X2; start = [0, 1]
# 2 equality constraints (root-finding)
# A(1): 10000*X1*X2 - 1 = 0
# B(1): exp(-X1) + exp(-X2) - 1.0001 = 0
function powellbs_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(0), T(1)])
    ExaModels.@add_con(c, T(10000)*x[1]*x[2] - T(1))
    ExaModels.@add_con(c, exp(-x[1]) + exp(-x[2]) - T(1.0001))
    return ExaModels.ExaModel(c; kwargs...)
end
