function waysea1ne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; start = T[1.0, 5.0])
    ExaModels.@add_con(c, x[1]^6 + x[2]^4 - T(17))
    ExaModels.@add_con(c, T(2) * x[1] + x[2] - T(4))
    return ExaModels.ExaModel(c; kwargs...)
end
