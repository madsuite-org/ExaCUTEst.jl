function eggcrate_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; start = T[1.0, 2.0])
    ExaModels.@add_obj(c, x[1]^2 + x[2]^2 + (T(5) * sin(x[1]))^2 + (T(5) * sin(x[2]))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
