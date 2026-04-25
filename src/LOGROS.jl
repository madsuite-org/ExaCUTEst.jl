# LOGROS: Logarithmic Rosenbrock variant (2 free vars, start=[-1.2,1.0])
# Element ROS(X,Y): F = 10000*(Y-X^2)^2 + (1-X)^2
# Group OBJ (LOG type): contribution = log(1 + F)
function logros_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T[T(-1.2), T(1.0)])
    ExaModels.@add_obj(c,
        log(T(1) + T(10000) * (x[2] - x[1]^2)^2 + (T(1) - x[1])^2))
    return ExaModels.ExaModel(c; kwargs...)
end
