function stregne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; start = T[-1.2, 1.0, 1e10, 1e10])
    ExaModels.@add_con(c, (x[2] - x[1]^2) / T(0.1))
    ExaModels.@add_con(c, x[1] - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
