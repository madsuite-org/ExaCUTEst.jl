function waysea2ne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; start = T[1.0, 5.0])
    ExaModels.@add_con(c, -T(4) * x[1]^2 - T(4) * x[2]^2 + T(2.5) * x[1] + T(13) * x[2] - T(9.340125))
    ExaModels.@add_con(c, x[2] - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
