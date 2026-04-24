function hs51_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf],
                       start = T[2.5, 0.5, 2.0, -1.0, 0.5])
    # SQ1M2: (x1-x2)²; SQ1P2: (x2+x3-2)²; SQM1: (x4-1)², (x5-1)²
    ExaModels.@add_obj(c,
        (x[1] - x[2])^2 + (x[2] + x[3] - T(2))^2 + (x[4] - T(1))^2 + (x[5] - T(1))^2)
    ExaModels.@add_con(c, x[1] + T(3)*x[2] - T(4))
    ExaModels.@add_con(c, x[3] + x[4] - T(2)*x[5])
    ExaModels.@add_con(c, x[2] - x[5])
    return ExaModels.ExaModel(c; kwargs...)
end
