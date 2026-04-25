function hs107_model(; T = Float64, backend = nothing, kwargs...)
    # CUTEst variable ordering: [X1, X2, X9, X8, X5, X6, X7, X4, X3] (SIF names)
    # FACT = 48.4/50.176 (SIF: RD FACT TMP 48.4 means FACT=48.4/TMP)
    C = T(48.4 / 50.176 * sin(0.25))
    D = T(48.4 / 50.176 * cos(0.25))
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 9;
                       lvar = T[0.0, 0.0, -Inf, -Inf, 0.90909, 0.90909, 0.90909, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, 1.09090, 1.09090, 1.09090, Inf, Inf],
                       start = T[0.8, 0.8, 0.0, 0.0, 1.0454, 1.0454, 1.0454, 0.2, 0.2])
    # OBJ: 3000*X1 + 2000*X2 + 1000*X1^3 + 666.667*X2^3
    ExaModels.@add_obj(c, T(3000)*x[1] + T(2000)*x[2] + T(1000)*x[1]^3 + T(666.667)*x[2]^3)
    # C1: -X1 + 0.4 + 2C*X5^2 - X5*X6*(D*sin(X8)+C*cos(X8)) - X5*X7*(D*sin(X9)+C*cos(X9))
    # CUTEst: x[4]=X8, x[3]=X9, x[5]=X5, x[6]=X6, x[7]=X7
    ExaModels.@add_con(c, -x[1] + T(0.4) + T(2)*C*x[5]^2
        - x[5]*x[6]*(D*sin(x[4]) + C*cos(x[4]))
        - x[5]*x[7]*(D*sin(x[3]) + C*cos(x[3])))
    # C2: -X2 + 0.4 + 2C*X6^2 + X5*X6*(D*sin(X8)-C*cos(X8)) + X6*X7*(D*sin(X8-X9)-C*cos(X8-X9))
    ExaModels.@add_con(c, -x[2] + T(0.4) + T(2)*C*x[6]^2
        + x[5]*x[6]*(D*sin(x[4]) - C*cos(x[4]))
        + x[6]*x[7]*(D*sin(x[4]-x[3]) - C*cos(x[4]-x[3])))
    # C3: 0.8 + 2C*X7^2 + X5*X7*(D*sin(X9)-C*cos(X9)) - X6*X7*(D*sin(X8-X9)+C*cos(X8-X9))
    ExaModels.@add_con(c, T(0.8) + T(2)*C*x[7]^2
        + x[5]*x[7]*(D*sin(x[3]) - C*cos(x[3]))
        - x[6]*x[7]*(D*sin(x[4]-x[3]) + C*cos(x[4]-x[3])))
    # C4: -X3 + 0.2 + 2D*X5^2 + X5*X6*(C*sin(X8)-D*cos(X8)) + X5*X7*(C*sin(X9)-D*cos(X9))
    # CUTEst: x[9]=X3
    ExaModels.@add_con(c, -x[9] + T(0.2) + T(2)*D*x[5]^2
        + x[5]*x[6]*(C*sin(x[4]) - D*cos(x[4]))
        + x[5]*x[7]*(C*sin(x[3]) - D*cos(x[3])))
    # C5: -X4 + 0.2 + 2D*X6^2 - X5*X6*(C*sin(X8)+D*cos(X8)) - X6*X7*(C*sin(X8-X9)+D*cos(X8-X9))
    # CUTEst: x[8]=X4
    ExaModels.@add_con(c, -x[8] + T(0.2) + T(2)*D*x[6]^2
        - x[5]*x[6]*(C*sin(x[4]) + D*cos(x[4]))
        - x[6]*x[7]*(C*sin(x[4]-x[3]) + D*cos(x[4]-x[3])))
    # C6: -0.337 + 2D*X7^2 - X5*X7*(C*sin(X9)+D*cos(X9)) + X6*X7*(C*sin(X8-X9)-D*cos(X8-X9))
    ExaModels.@add_con(c, T(-0.337) + T(2)*D*x[7]^2
        - x[5]*x[7]*(C*sin(x[3]) + D*cos(x[3]))
        + x[6]*x[7]*(C*sin(x[4]-x[3]) - D*cos(x[4]-x[3])))
    return ExaModels.ExaModel(c; kwargs...)
end
