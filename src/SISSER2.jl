function sisser2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; start = T[1.0, 0.1])
    ExaModels.@add_obj(c, T(3) * x[1]^4 - T(2) * (x[1] * x[2])^2 + T(3) * x[2]^4)
    return ExaModels.ExaModel(c; kwargs...)
end
