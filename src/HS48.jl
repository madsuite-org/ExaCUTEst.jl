function hs48_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf],
                       start = T[3.0, 5.0, -3.0, 2.0, -2.0])
    ExaModels.@add_obj(c,
        (x[1] - T(1))^2 + (x[2] - x[3])^2 + (x[4] - x[5])^2)
    ExaModels.@add_con(c, x[1] + x[2] + x[3] + x[4] + x[5] - T(5))
    ExaModels.@add_con(c, x[3] - T(2)*x[4] - T(2)*x[5] + T(3))
    return ExaModels.ExaModel(c; kwargs...)
end
