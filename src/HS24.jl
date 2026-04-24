function hs24_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[0.0, 0.0], uvar = T[Inf, Inf],
                       start = T[1.0, 0.5])
    # OBJ: ((X1-3)²-9)*X2³/(27√3)
    ExaModels.@add_obj(c, ((x[1] - T(3))^2 - T(9)) * x[2]^3 / (T(27) * sqrt(T(3))))
    # CON1 (G): X1/√3 - X2 ≥ 0
    ExaModels.@add_con(c, x[1] / sqrt(T(3)) - x[2]; lcon = T(0), ucon = T(Inf))
    # CON2 (G): X1 + √3*X2 ≥ 0
    ExaModels.@add_con(c, x[1] + sqrt(T(3)) * x[2]; lcon = T(0), ucon = T(Inf))
    # CON3 (G): 6 - X1 - √3*X2 ≥ 0
    ExaModels.@add_con(c, T(6) - x[1] - sqrt(T(3)) * x[2]; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
