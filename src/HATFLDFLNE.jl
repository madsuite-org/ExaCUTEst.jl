function hatfldflne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; start = T[1.2, -1.2, 0.98])
    ExaModels.@add_con(c, x[1] + x[2] * x[3] - T(0.032))
    ExaModels.@add_con(c, x[1] + x[2] * x[3]^2 - T(0.056))
    ExaModels.@add_con(c, x[1] + x[2] * x[3]^3 - T(0.099))
    return ExaModels.ExaModel(c; kwargs...)
end
