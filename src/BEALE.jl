function beale_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf], start = T[1.0, 1.0])
    ExaModels.@add_obj(c, (x[1] * (T(1) - x[2]) - T(1.5))^2)
    ExaModels.@add_obj(c, (x[1] * (T(1) - x[2]^2) - T(2.25))^2)
    ExaModels.@add_obj(c, (x[1] * (T(1) - x[2]^3) - T(2.625))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
