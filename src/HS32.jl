function hs32_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[0.0, 0.0, 0.0], uvar = T[Inf, Inf, Inf],
                       start = T[0.1, 0.7, 0.2])
    # OBJ: (X1+3X2+X3)²+4*(X1-X2)²
    ExaModels.@add_obj(c, (x[1] + T(3)*x[2] + x[3])^2 + T(4)*(x[1] - x[2])^2)
    # CUTEst order: C2 (E, linear) before C1 (G, nonlinear)
    # C2 (E): -X1-X2-X3+1 = 0 (SIF group value: coefficients -1,-1,-1, constant -1)
    ExaModels.@add_con(c, -x[1] - x[2] - x[3] + T(1))
    # C1 (G): 6X2+4X3-X1³-3 ≥ 0
    ExaModels.@add_con(c, T(6)*x[2] + T(4)*x[3] - x[1]^3 - T(3); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
