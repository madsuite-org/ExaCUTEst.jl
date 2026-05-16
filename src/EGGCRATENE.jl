function eggcratene_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; start = T[1.0, 2.0])
    ExaModels.@add_con(c, x[1])
    ExaModels.@add_con(c, x[2])
    ExaModels.@add_con(c, T(5) * sin(x[1]))
    ExaModels.@add_con(c, T(5) * sin(x[2]))
    return ExaModels.ExaModel(c; kwargs...)
end
