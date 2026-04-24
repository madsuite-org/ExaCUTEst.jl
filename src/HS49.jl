function hs49_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf],
                       start = T[10.0, 7.0, 2.0, -3.0, 0.8])
    ExaModels.@add_obj(c,
        (x[1] - x[2])^2 + (x[3] - T(1))^2 + (x[4] - T(1))^4 + (x[5] - T(1))^6)
    ExaModels.@add_con(c, x[1] + x[2] + x[3] + T(4)*x[4] - T(7))
    ExaModels.@add_con(c, x[3] + T(5)*x[5] - T(6))
    return ExaModels.ExaModel(c; kwargs...)
end
