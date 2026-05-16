function hatfldfls_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; start = T(0.1))
    ExaModels.@add_obj(c, (x[1] + x[2] * x[3] - T(0.032))^2)
    ExaModels.@add_obj(c, (x[1] + x[2] * x[3]^2 - T(0.056))^2)
    ExaModels.@add_obj(c, (x[1] + x[2] * x[3]^3 - T(0.099))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
