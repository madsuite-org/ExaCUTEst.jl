function hs40_model(; T = Float64, backend = nothing, kwargs...)
    # CUTEst ordering: x[1]=X1, x[2]=X2, x[3]=X4, x[4]=X3
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T[-Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf],
                       start = T[0.8, 0.8, 0.8, 0.8])
    ExaModels.@add_obj(c, -x[1] * x[2] * x[3] * x[4])
    # CON1 (E): X1^3 + X2^2 - 1 = x[1]^3 + x[2]^2 - 1 = 0
    ExaModels.@add_con(c, x[1]^3 + x[2]^2 - T(1))
    # CON2 (E): -X3 + X1^2*X4 = -x[4] + x[1]^2*x[3] = 0
    ExaModels.@add_con(c, -x[4] + x[1]^2 * x[3])
    # CON3 (E): -X2 + X4^2 = -x[2] + x[3]^2 = 0
    ExaModels.@add_con(c, -x[2] + x[3]^2)
    return ExaModels.ExaModel(c; kwargs...)
end
