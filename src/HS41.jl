function hs41_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T[0.0, 0.0, 0.0, 0.0],
                       uvar = T[1.0, 1.0, 1.0, 2.0],
                       start = T[2.0, 2.0, 2.0, 2.0])
    ExaModels.@add_obj(c, T(2) - x[1] * x[2] * x[3])
    # CON1 (E, linear): x1 + 2x2 + 2x3 - x4 = 0
    ExaModels.@add_con(c, x[1] + T(2)*x[2] + T(2)*x[3] - x[4])
    return ExaModels.ExaModel(c; kwargs...)
end
