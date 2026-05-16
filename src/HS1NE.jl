function hs1ne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -1.5], uvar = T[Inf, Inf],
                       start = T[-2.0, 1.0])
    ExaModels.@add_con(c, (x[2] - x[1]^2) / T(0.1))
    ExaModels.@add_con(c, x[1] - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
