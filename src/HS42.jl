function hs42_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # All variables free; CUTEst orders them as X4, X3, X2, X1 (DO loop reversed)
    ExaModels.@add_var(c, x, 4; lvar = T(-Inf), uvar = T(Inf),
                       start = T[1.0, 1.0, 1.0, 1.0])
    # OBJ: Σ (X(i) - i)^2 for i=1..4; CUTEst maps x[1]↔X4, x[2]↔X3, x[3]↔X2, x[4]↔X1
    ExaModels.@add_obj(c,
        (x[1] - T(4))^2 + (x[2] - T(3))^2 + (x[3] - T(2))^2 + (x[4] - T(1))^2)
    # CON1 (E): X1 - 2 = 0  →  x[4] - 2 = 0
    ExaModels.@add_con(c, x[4] - T(2))
    # CON2 (E): X3^2 + X4^2 - 2 = 0  →  x[2]^2 + x[1]^2 - 2 = 0
    ExaModels.@add_con(c, x[2]^2 + x[1]^2 - T(2))
    return ExaModels.ExaModel(c; kwargs...)
end
