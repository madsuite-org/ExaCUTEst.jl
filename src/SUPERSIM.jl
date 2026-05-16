# SUPERSIM: Super simple LP in 2 variables
# min x
# s.t. x + 2*y = 2
#      2*x + y = 2
# Bounds: x >= 0 (default), y free (XR)
function supersim_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[0.0, -Inf], uvar = T[Inf, Inf])
    ExaModels.@add_obj(c, x[1])
    ExaModels.@add_con(c, x[1] + T(2) * x[2] - T(2))
    ExaModels.@add_con(c, T(2) * x[1] + x[2] - T(2))
    return ExaModels.ExaModel(c; kwargs...)
end
