function hs39_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T[-Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf],
                       start = T[2.0, 2.0, 2.0, 2.0])
    ExaModels.@add_obj(c, -x[1])
    # CON1 (E): x2 - x1^3 - x3^2 = 0
    ExaModels.@add_con(c, x[2] - x[1]^3 - x[3]^2)
    # CON2 (E): -x2 + x1^2 - x4^2 = 0
    ExaModels.@add_con(c, -x[2] + x[1]^2 - x[4]^2)
    return ExaModels.ExaModel(c; kwargs...)
end
