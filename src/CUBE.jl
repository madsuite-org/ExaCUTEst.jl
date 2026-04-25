function cube_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf], start = T[-1.2, 1.0])
    ExaModels.@add_obj(c, (x[1] - T(1))^2 + T(100) * (x[2] - x[1]^3)^2)
    return ExaModels.ExaModel(c; kwargs...)
end
