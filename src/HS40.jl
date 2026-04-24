function hs40_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T[-Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf],
                       start = T[0.8, 0.8, 0.8, 0.8])
    ExaModels.@add_obj(c, -x[1] * x[2] * x[3] * x[4])
    # CON1 (E): x1^3 + x2^2 - 1 = 0
    ExaModels.@add_con(c, x[1]^3 + x[2]^2 - T(1))
    # CON2 (E): -x3 + x1^2 * x4 = 0
    ExaModels.@add_con(c, -x[3] + x[1]^2 * x[4])
    # CON3 (E): -x2 + x4^2 = 0
    ExaModels.@add_con(c, -x[2] + x[4]^2)
    return ExaModels.ExaModel(c; kwargs...)
end
