function ssebnln_model(; T = Float64, backend = nothing, kwargs...)
    # SSEBNLN: Power Generation QP, classification LQR2-RN-194-96
    # DAYS=1, HOURS=24. CUTEst variable ordering determined from Jacobian analysis.
    #
    # Variable positions (1-based) per IH=1..24:
    # For IH=1..6: large positions (descending from ~184), QG/QP in positions 7..48
    # For IH=7..24: smaller positions (ascending from 49), QG/QP in positions 1..46 (descending)
    # V(0,24)=192, R(0,24)=191 (fixed)

    Z_val = T(517.0)
    D_data = T[578.0, 517.0, 461.0, 369.0, 299.0, 269.0, 370.0, 559.0,
               689.0, 728.0, 683.0, 626.0, 619.0, 586.0, 582.0, 625.0,
               821.0, 883.0, 768.0, 711.0, 677.0, 630.0, 545.0, 565.0]

    HOURS = 24

    # Variable position tables (from CUTEst Jacobian)
    P1i = [184,160,136,112,88,64,51,59,67,75,83,91,99,107,115,123,131,139,147,155,163,171,179,187]
    P2i = [183,159,135,111,87,63,52,60,68,76,84,92,100,108,116,124,132,140,148,156,164,172,180,188]
    QHi = [176,152,128,104,80,56,53,61,69,77,85,93,101,109,117,125,133,141,149,157,165,173,181,189]
    Si  = [175,151,127,103,79,55,54,62,70,78,86,94,102,110,118,126,134,142,150,158,166,174,182,190]
    Vi  = [168,144,120,96,72,49,57,65,73,81,89,97,105,113,121,129,137,145,153,161,169,177,185,193]
    Ri  = [167,143,119,95,71,50,58,66,74,82,90,98,106,114,122,130,138,146,154,162,170,178,186,194]
    QGi = [7,15,23,31,39,47,46,44,42,38,36,34,30,28,26,22,20,18,14,12,10,6,4,2]
    QPi = [8,16,24,32,40,48,45,43,41,37,35,33,29,27,25,21,19,17,13,11,9,5,3,1]
    V0  = 192  # V(0,24) fixed
    R0  = 191  # R(0,24) fixed

    lvar = fill(T(-Inf), 194)
    uvar = fill(T(Inf), 194)
    x0   = fill(T(0), 194)

    lvar[V0] = T(240000); uvar[V0] = T(240000); x0[V0] = T(240000)
    lvar[R0] = T(3500);   uvar[R0] = T(3500);   x0[R0] = T(3500)

    for ih in 1:HOURS
        lvar[P1i[ih]] = T(70);     uvar[P1i[ih]] = T(325);    x0[P1i[ih]] = T(70)
        lvar[P2i[ih]] = T(90);     uvar[P2i[ih]] = T(290);    x0[P2i[ih]] = T(90)
        lvar[QHi[ih]] = T(25);     uvar[QHi[ih]] = T(500);    x0[QHi[ih]] = T(25)
        lvar[Si[ih]]  = T(0);      uvar[Si[ih]]  = T(Inf);    x0[Si[ih]]  = T(0)
        lvar[Vi[ih]]  = T(180000); uvar[Vi[ih]]  = T(280000); x0[Vi[ih]]  = T(240000)
        lvar[Ri[ih]]  = T(0);      uvar[Ri[ih]]  = T(6000);   x0[Ri[ih]]  = T(3500)
        lvar[QGi[ih]] = T(0);      uvar[QGi[ih]] = T(300);    x0[QGi[ih]] = T(0)
        lvar[QPi[ih]] = T(0);      uvar[QPi[ih]] = T(225);    x0[QPi[ih]] = T(225)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 194; lvar = lvar, uvar = uvar, start = x0)

    # Objective: sum over IH of (1000*P1 + 1500*P2 + 1200*QH + 1200*S + 1200*QG - 1200*QP)
    for ih in 1:HOURS
        ExaModels.@add_obj(c, T(1000)*x[P1i[ih]] + T(1500)*x[P2i[ih]]
                            + T(1200)*x[QHi[ih]] + T(1200)*x[Si[ih]]
                            + T(1200)*x[QGi[ih]] - T(1200)*x[QPi[ih]])
    end

    # H constraints (water balance) - IH=1..24 order
    ExaModels.@add_con(c, x[Vi[1]] + x[Si[1]] + x[QHi[1]] - x[V0] - Z_val)
    for ih in 2:HOURS
        ExaModels.@add_con(c, x[Vi[ih]] + x[Si[ih]] + x[QHi[ih]] - x[Vi[ih-1]] - Z_val)
    end

    # R constraints (storage balance) - IH=1..24 order
    ExaModels.@add_con(c, x[Ri[1]] + x[QGi[1]] - x[QPi[1]] - x[R0] - T(0.01*517.0))
    for ih in 2:HOURS
        ExaModels.@add_con(c, x[Ri[ih]] + x[QGi[ih]] - x[QPi[ih]] - x[Ri[ih-1]] - T(0.01*517.0))
    end

    # D constraints (demand) - IH=1..24 order, G-type (≥ D_val)
    for ih in 1:HOURS
        ExaModels.@add_con(c, x[P1i[ih]] + x[P2i[ih]] + x[QHi[ih]]
                            + x[QGi[ih]] - T(1.33)*x[QPi[ih]] - T(D_data[ih]);
                           lcon = T(0), ucon = T(Inf))
    end

    # QG*QP constraints - IH=1..24 order
    for ih in 1:HOURS
        ExaModels.@add_con(c, x[QGi[ih]] * x[QPi[ih]])
    end

    return ExaModels.ExaModel(c; kwargs...)
end
