function denschndne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; start = T(10.0))
    ExaModels.@add_con(c, x[1]^2 + x[2]^3 - x[3]^4)
    ExaModels.@add_con(c, T(2) * x[1] * x[2] * x[3])
    ExaModels.@add_con(c, T(2) * x[1] * x[2] - T(3) * x[2] * x[3] + x[1] * x[3])
    return ExaModels.ExaModel(c; kwargs...)
end
