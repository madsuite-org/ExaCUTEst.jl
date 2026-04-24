function hs35_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[0.0, 0.0, 0.0], uvar = T[Inf, Inf, Inf],
                       start = T[0.5, 0.5, 0.5])
    # OBJ: 9+2x1²+2x2²+x3²+2x1x2+2x1x3-8x1-6x2-4x3
    ExaModels.@add_obj(c, T(9) - T(8)*x[1] - T(6)*x[2] - T(4)*x[3] + T(2)*x[1]^2 + T(2)*x[2]^2 +
                          x[3]^2 + T(2)*x[1]*x[2] + T(2)*x[1]*x[3])
    # CON1 (G, linear): 3-x[1]-x[2]-2x[3] ≥ 0
    ExaModels.@add_con(c, T(3) - x[1] - x[2] - T(2)*x[3]; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
