function hs3_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, 0.0], uvar = T[Inf, Inf],
                       start = T[10.0, 1.0])
    ExaModels.@add_obj(c, x[2] + (x[2] - x[1])^2 / T(100000))
    return ExaModels.ExaModel(c; kwargs...)
end
