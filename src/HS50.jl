function hs50_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf],
                       start = T[35.0, -31.0, 11.0, 5.0, -5.0])
    ExaModels.@add_obj(c,
        (x[1] - x[2])^2 + (x[2] - x[3])^2 + (x[3] - x[4])^4 + (x[4] - x[5])^2)
    ExaModels.@add_con(c, x[1] + T(2)*x[2] + T(3)*x[3] - T(6))
    ExaModels.@add_con(c, x[2] + T(2)*x[3] + T(3)*x[4] - T(6))
    ExaModels.@add_con(c, x[3] + T(2)*x[4] + T(3)*x[5] - T(6))
    return ExaModels.ExaModel(c; kwargs...)
end
