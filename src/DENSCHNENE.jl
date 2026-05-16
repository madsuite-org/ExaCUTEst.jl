function denschnene_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; start = T[2.0, 3.0, -8.0])
    ExaModels.@add_con(c, x[1])
    ExaModels.@add_con(c, x[2] + x[2]^2)
    ExaModels.@add_con(c, exp(x[3]) - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
