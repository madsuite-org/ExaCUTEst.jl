function hs109_model(; T = Float64, backend = nothing, kwargs...)
    # Problem parameters
    A  = T(50.176)
    B  = T(sin(0.25))                # RF B SIN 0.25
    C  = T(cos(0.25))                # RF C COS 0.25
    D  = T(0.55)
    # Derived parameters (RM)
    p2B    = T(2.0) * B              # 2B
    pm2C   = T(-2.0) * C            # -2C
    p0007A = A * T(0.0007533)       # .0007+A  = A * 0.0007533
    p400A  = A * T(400.0)           # 400A (= -(-400A))
    p200A  = A * T(200.0)           # 200A
    p881A  = A * T(881.779)         # -(-881+A)  = -A*(-881.779) = A*881.779
    p22A   = A * T(22.938)          # 22+A  = A*22.938
    E_val  = T(2250000.0)           # -(-E) where RE -E -2250000

    # Combined coefficient for SQ(X5/X6/X7): (-2C + .0007+A)
    coef57 = pm2C + p0007A

    c = ExaModels.ExaCore(T; backend, concrete = Val(true))

    # Variables X1..X9 in SIF order = CUTEst order (grad=[3,2,0,...] confirms X1=x[1])
    ExaModels.@add_var(c, x, 9;
                       lvar = T[0.0, 0.0, -0.55, -0.55, 196.0, 196.0, 196.0, -400.0, -400.0],
                       uvar = T[Inf, Inf,  0.55,  0.55, 252.0, 252.0, 252.0,  800.0,  800.0],
                       start = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Objective: 3*X1 + 2*X2 + 1e-6*X1^3 + 0.522074e-6*X2^3
    ExaModels.@add_obj(c, T(3)*x[1] + T(2)*x[2] + T(1e-6)*x[1]^3 + T(0.522074e-6)*x[2]^3)

    # ── Constraint ordering (CUTEst empirical ordering) ────────────────────────
    # cons[1] (G): C1 = X4 - X3 + D  >= 0
    ExaModels.@add_con(c, x[4] - x[3] + D; lcon = T(0), ucon = T(Inf))

    # cons[2] (G): C2 = X3 - X4 + D  >= 0
    ExaModels.@add_con(c, x[3] - x[4] + D; lcon = T(0), ucon = T(Inf))

    # cons[3] (E): C10 = X5*X7*cos(X4-0.25) + X6*X7*cos(X4-X3-0.25)
    #                    + coef57*X7^2 - p22A  = 0
    ExaModels.@add_con(c, x[5]*x[7]*cos(x[4] - T(0.25))
                        + x[6]*x[7]*cos(x[4] - x[3] - T(0.25))
                        + coef57*x[7]^2 - p22A)

    # cons[4] (E): C9 = A*X9 + X5*X6*cos(X3-0.25) + X6*X7*cos(X3-X4-0.25)
    #                   + coef57*X6^2 - p200A  = 0
    ExaModels.@add_con(c, A*x[9]
                        + x[5]*x[6]*cos(x[3] - T(0.25))
                        + x[6]*x[7]*cos(x[3] - x[4] - T(0.25))
                        + coef57*x[6]^2 - p200A)

    # cons[5] (E): C5 = -A*X1 + X5*X6*sin(-X3-0.25) + X5*X7*sin(-X4-0.25)
    #                   + p2B*X5^2 + p400A  = 0
    ExaModels.@add_con(c, -A*x[1]
                        + x[5]*x[6]*sin(-x[3] - T(0.25))
                        + x[5]*x[7]*sin(-x[4] - T(0.25))
                        + p2B*x[5]^2 + p400A)

    # cons[6] (E): C6 = -A*X2 + X5*X6*sin(X3-0.25) + X6*X7*sin(X3-X4-0.25)
    #                   + p2B*X6^2 + p400A  = 0
    ExaModels.@add_con(c, -A*x[2]
                        + x[5]*x[6]*sin(x[3] - T(0.25))
                        + x[6]*x[7]*sin(x[3] - x[4] - T(0.25))
                        + p2B*x[6]^2 + p400A)

    # cons[7] (E): C7 = X5*X7*sin(X4-0.25) + X6*X7*sin(X4-X3-0.25)
    #                   + p2B*X7^2 + p881A  = 0
    ExaModels.@add_con(c, x[5]*x[7]*sin(x[4] - T(0.25))
                        + x[6]*x[7]*sin(x[4] - x[3] - T(0.25))
                        + p2B*x[7]^2 + p881A)

    # cons[8] (E): C8 = A*X8 + X5*X6*cos(-X3-0.25) + X5*X7*cos(-X4-0.25)
    #                   + coef57*X5^2 - p200A  = 0
    ExaModels.@add_con(c, A*x[8]
                        + x[5]*x[6]*cos(-x[3] - T(0.25))
                        + x[5]*x[7]*cos(-x[4] - T(0.25))
                        + coef57*x[5]^2 - p200A)

    # cons[9] (G): C3 = -X1^2 - X8^2 + 2250000  >= 0
    ExaModels.@add_con(c, -x[1]^2 - x[8]^2 + E_val; lcon = T(0), ucon = T(Inf))

    # cons[10] (G): C4 = -X2^2 - X9^2 + 2250000  >= 0
    ExaModels.@add_con(c, -x[2]^2 - x[9]^2 + E_val; lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
