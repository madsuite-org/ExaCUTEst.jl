function sseblin_model(; T = Float64, backend = nothing, kwargs...)
    # SSEBLIN: Power Generation LP, classification LLR2-MN-194-72
    # DAYS=1, HOURS=24, PERIOD=24
    # Variables (in SIF declaration order):
    #   V(0,24), R(0,24)                         [2 fixed vars]
    #   for ID=1..1, IH=1..24: P1,P2,QH,S,QG,QP,V,R  [8*24=192 vars]
    # Total: 194 variables

    # Data: Z(day) influx, D(day,hour) demand
    Z = T[517.0, 400.0, 1017.0, 667.0, 600.0, 421.0, 425.0]
    D = T[
        578.0, 517.0, 461.0, 369.0, 299.0, 269.0, 370.0, 559.0,
        689.0, 728.0, 683.0, 626.0, 619.0, 586.0, 582.0, 625.0,
        821.0, 883.0, 768.0, 711.0, 677.0, 630.0, 545.0, 565.0,
    ]  # D(1,1..24)

    # Variable index layout (1-based):
    #   1  = V(0,24)   [fixed=240000]
    #   2  = R(0,24)   [fixed=3500]
    # Then for ID=1, IH=1..24: P1,P2,QH,S,QG,QP,V,R
    #   P1(1,IH) at index 2 + (IH-1)*8 + 1
    #   P2(1,IH) at index 2 + (IH-1)*8 + 2
    #   QH(1,IH) at index 2 + (IH-1)*8 + 3
    #   S(1,IH)  at index 2 + (IH-1)*8 + 4
    #   QG(1,IH) at index 2 + (IH-1)*8 + 5
    #   QP(1,IH) at index 2 + (IH-1)*8 + 6
    #   V(1,IH)  at index 2 + (IH-1)*8 + 7
    #   R(1,IH)  at index 2 + (IH-1)*8 + 8

    # For DAYS=1: the "day 0" wrap is V(0,HOURS)=V(0,24)=x[1], R(0,HOURS)=R(0,24)=x[2]
    # H(1,1): V(1,1)+S(1,1)+QH(1,1)-V(0,24) = Z(1)
    # H(1,IH) for IH=2..24: V(1,IH)+S(1,IH)+QH(1,IH)-V(1,IH-1) = Z(1)
    # R(1,1): R(1,1)+QG(1,1)-QP(1,1)-R(0,24) = 0.01*Z(1)
    # R(1,IH) for IH=2..24: R(1,IH)+QG(1,IH)-QP(1,IH)-R(1,IH-1) = 0.01*Z(1)
    # D(1,IH): P1(1,IH)+P2(1,IH)+QH(1,IH)+QG(1,IH)-1.33*QP(1,IH) >= D(1,IH)

    HOURS = 24

    # lvar/uvar arrays: 194 variables
    lvar = fill(T(-Inf), 194)
    uvar = fill(T(Inf), 194)
    x0   = fill(T(0), 194)

    # V(0,24) = x[1]: fixed at 240000
    lvar[1] = T(240000)
    uvar[1] = T(240000)
    x0[1]   = T(240000)
    # R(0,24) = x[2]: fixed at 3500
    lvar[2] = T(3500)
    uvar[2] = T(3500)
    x0[2]   = T(3500)

    for ih in 1:HOURS
        base = 2 + (ih-1)*8
        # P1: [70, 325]
        lvar[base+1] = T(70); uvar[base+1] = T(325); x0[base+1] = T(70)
        # P2: [90, 290]
        lvar[base+2] = T(90); uvar[base+2] = T(290); x0[base+2] = T(90)
        # QH: [25, 500]
        lvar[base+3] = T(25); uvar[base+3] = T(500); x0[base+3] = T(25)
        # S: [0, Inf]
        lvar[base+4] = T(0);  uvar[base+4] = T(Inf); x0[base+4] = T(0)
        # QG: [0, 300]
        lvar[base+5] = T(0);  uvar[base+5] = T(300); x0[base+5] = T(0)
        # QP: [0, 225]
        lvar[base+6] = T(0);  uvar[base+6] = T(225); x0[base+6] = T(225)
        # V: [180000, 280000]
        lvar[base+7] = T(180000); uvar[base+7] = T(280000); x0[base+7] = T(240000)
        # R: [0, 6000]
        lvar[base+8] = T(0);  uvar[base+8] = T(6000); x0[base+8] = T(3500)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 194; lvar = lvar, uvar = uvar, start = x0)

    # Objective: sum over IH of (1000*P1 + 1500*P2 + 1200*QH + 1200*S + 1200*QG - 1200*QP)
    ExaModels.@add_obj(c, (1000*x[2+(i-1)*8+1] + 1500*x[2+(i-1)*8+2]
                         + 1200*x[2+(i-1)*8+3] + 1200*x[2+(i-1)*8+4]
                         + 1200*x[2+(i-1)*8+5] - 1200*x[2+(i-1)*8+6])
                       for i in 1:HOURS)

    # Hydro-power constraints H(1,1): V(1,1)+S(1,1)+QH(1,1)-V(0,24) = Z(1)
    # V(1,1)=x[9], S(1,1)=x[6], QH(1,1)=x[5], V(0,24)=x[1]
    ExaModels.@add_con(c, x[9] + x[6] + x[5] - x[1] - T(Z[1]))

    # H(1,IH) for IH=2..24: V(1,IH)+S(1,IH)+QH(1,IH)-V(1,IH-1) = Z(1)
    ExaModels.@add_con(c, x[2+i*8+7] + x[2+i*8+4] + x[2+i*8+3] - x[2+(i-1)*8+7] - T(Z[1])
                       for i in 1:HOURS-1)

    # Reservoir constraints R(1,1): R(1,1)+QG(1,1)-QP(1,1)-R(0,24) = 0.01*Z(1)
    # R(1,1)=x[10], QG(1,1)=x[7], QP(1,1)=x[8], R(0,24)=x[2]
    ExaModels.@add_con(c, x[10] + x[7] - x[8] - x[2] - T(0.01*Z[1]))

    # R(1,IH) for IH=2..24: R(1,IH)+QG(1,IH)-QP(1,IH)-R(1,IH-1) = 0.01*Z(1)
    ExaModels.@add_con(c, x[2+i*8+8] + x[2+i*8+5] - x[2+i*8+6] - x[2+(i-1)*8+8] - T(0.01*Z[1])
                       for i in 1:HOURS-1)

    # Demand constraints D(1,IH): P1+P2+QH+QG-1.33*QP >= D(1,IH)
    # Encode as: P1+P2+QH+QG-1.33*QP - D[IH] >= 0  →  lcon=0, ucon=Inf
    ExaModels.@add_par(c, D_par, D)
    ExaModels.@add_con(c, x[2+(i-1)*8+1] + x[2+(i-1)*8+2] + x[2+(i-1)*8+3]
                        + x[2+(i-1)*8+5] - T(1.33)*x[2+(i-1)*8+6] - D_par[i]
                       for i in 1:HOURS;
                       lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
