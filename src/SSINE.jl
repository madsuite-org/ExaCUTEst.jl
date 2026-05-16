function ssine_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; start = ones(T, 3))
    ExaModels.@add_con(c, x[1]^2 * x[3] - T(4))
    ExaModels.@add_con(c, x[2]^2 + x[3])
    return ExaModels.ExaModel(c; kwargs...)
end
