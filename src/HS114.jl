function hs114_model(; T = Float64, backend = nothing, kwargs...)
    # Problem parameters
    A    = T(0.99)
    B    = T(0.9)
    INVA = T(1.0 / 0.99)    # RD INVA A 1.0
    INVB = T(1.0 / 0.9)     # RD INVB B 1.0

    c = ExaModels.ExaCore(T; backend, concrete = Val(true))

    # CUTEst variable ordering: x[1]=X7, x[2]=X4, x[3]=X3, x[4]=X2, x[5]=X5,
    #                            x[6]=X9, x[7]=X1, x[8]=X8, x[9]=X6, x[10]=X10
    ExaModels.@add_var(c, x, 10;
                       lvar  = T[90.0, 1e-5, 1e-5, 1e-5, 1e-5, 1.2, 1e-5, 3.0, 85.0, 145.0],
                       uvar  = T[95.0, 5000.0, 120.0, 16000.0, 2000.0, 4.0, 2000.0, 12.0, 93.0, 162.0],
                       start = T[92.8, 3048.0, 110.0, 12000.0, 1974.0, 3.6, 1745.0, 8.0, 89.2, 145.0])

    # CUTEst names: x[1]=X7, x[2]=X4, x[3]=X3, x[4]=X2, x[5]=X5,
    #               x[6]=X9, x[7]=X1, x[8]=X8, x[9]=X6, x[10]=X10
    # Objective: X2*0.035 + X3*10 + X5*3.36 + X1*5.04 + OE1 (-0.063*X4*X7)
    ExaModels.@add_obj(c, T(0.035)*x[4] + T(10.0)*x[3] + T(3.36)*x[5]
                        + T(5.04)*x[7] + T(-0.063)*x[2]*x[1])

    # ── Constraint ordering (CUTEst empirical ordering) ────────────────────────
    # cons[1] (E): C9 = 1.22*X4 - X1 - X5 = 0
    ExaModels.@add_con(c, T(1.22)*x[2] - x[7] - x[5])

    # cons[2] (G): C2 = 3*X7 - A*X10 - 133  >= 0
    ExaModels.@add_con(c, T(3.0)*x[1] - A*x[10] - T(133.0); lcon = T(0), ucon = T(Inf))

    # cons[3] (G): C3 = 0.222*X10 + INVB*X9 - 35.82  >= 0
    ExaModels.@add_con(c, T(0.222)*x[10] + INVB*x[6] - T(35.82); lcon = T(0), ucon = T(Inf))

    # cons[4] (G): C4 = INVA*X10 - 3*X7 + 133  >= 0
    ExaModels.@add_con(c, INVA*x[10] - T(3.0)*x[1] + T(133.0); lcon = T(0), ucon = T(Inf))

    # cons[5] (G): C1 = -0.222*X10 - B*X9 + 35.82  >= 0
    ExaModels.@add_con(c, T(-0.222)*x[10] - B*x[6] + T(35.82); lcon = T(0), ucon = T(Inf))

    # cons[6] (E): C11 = (X2+X5)/X1 - X8 = 0
    #   RAP2 element with IV: SUM = X2+X5, SUMZ = X1 → F = SUM/SUMZ
    ExaModels.@add_con(c, (x[4] + x[5])/x[7] - x[8])

    # cons[7] (E): C10 = 98000*X3/(X4*X9+1000*X3) - X6 = 0
    #   RAP1 element: W*X3/(X4*X9+1000*X3) with W=98000, X=X3, S=X4, T=X9, U=X3
    ExaModels.@add_con(c, T(98000.0)*x[3]/(x[2]*x[6] + T(1000.0)*x[3]) - x[9])

    # cons[8] (G): C8 = -1.098*X8 - 0.325*X6 + INVA*X7 + 0.038*X8^2 - 57.425  >= 0
    ExaModels.@add_con(c, T(-1.098)*x[8] - T(0.325)*x[9] + INVA*x[1]
                        + T(0.038)*x[8]^2 - T(57.425); lcon = T(0), ucon = T(Inf))

    # cons[9] (G): C5 = 1.12*X1 - A*X4 + 0.13167*X1*X8 - 0.00667*X1*X8^2  >= 0
    ExaModels.@add_con(c, T(1.12)*x[7] - A*x[2]
                        + T(0.13167)*x[7]*x[8] - T(0.00667)*x[7]*x[8]^2;
                       lcon = T(0), ucon = T(Inf))

    # cons[10] (G): C7 = -1.12*X1 + INVA*X4 - 0.13167*X1*X8 + 0.00667*X1*X8^2  >= 0
    ExaModels.@add_con(c, T(-1.12)*x[7] + INVA*x[2]
                        - T(0.13167)*x[7]*x[8] + T(0.00667)*x[7]*x[8]^2;
                       lcon = T(0), ucon = T(Inf))

    # cons[11] (G): C6 = 1.098*X8 + 0.325*X6 - A*X7 - 0.038*X8^2 + 57.425  >= 0
    ExaModels.@add_con(c, T(1.098)*x[8] + T(0.325)*x[9] - A*x[1]
                        - T(0.038)*x[8]^2 + T(57.425); lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
