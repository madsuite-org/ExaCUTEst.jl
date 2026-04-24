function hs38_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T[-10.0, -10.0, -10.0, -10.0],
                       uvar = T[10.0, 10.0, 10.0, 10.0],
                       start = T[-3.0, -1.0, -3.0, -1.0])
    # Colville function (Powell's bad-scaled)
    # G(1-4) L2 with constants=1; G2,G4 scale=1/10.1; G5 L2 scale=0.01; G6 L2 scale=1/90
    # G7 identity element PROD(X2,X4)=(1-X2)*(1-X4) coeff=19.8
    ExaModels.@add_obj(c,
        (x[1] - T(1))^2 + T(10.1) * (x[2] - T(1))^2 +
        (x[3] - T(1))^2 + T(10.1) * (x[4] - T(1))^2 +
        T(100) * (x[2] - x[1]^2)^2 + T(90) * (x[4] - x[3]^2)^2 +
        T(19.8) * (T(1) - x[2]) * (T(1) - x[4]))
    return ExaModels.ExaModel(c; kwargs...)
end
