function streg_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; start = T[-1.2, 1.0, 1e10, 1e10])
    ExaModels.@add_obj(c, (x[2] - x[1]^2)^2 / T(0.01) + (x[1] - T(1))^2 + T(0.5) * x[3]^2 + T(0.5) * x[4]^2)
    return ExaModels.ExaModel(c; kwargs...)
end
