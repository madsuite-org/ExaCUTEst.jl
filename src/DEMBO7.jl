# DEMBO7: 7-stage membrane separation model (Dembo 1976)
# 16 variables, 20 constraints (1 G-range + 19 L)
# Element types: INV=1/X, 2PR=X*Y, QT=X/Y, QTQT=X*W/(Y*Z), 2PRRC=X*Y/Z, QTRC=X/(Y*Z), SQQT=X²/Y
function dembo7_model(; T = Float64, backend = nothing, kwargs...)
    # Variable indices: X(I) = I for I=1..16
    # Bounds (DEFAULT: 0.1 <= X(I) <= 0.9 for most)
    lvar = fill(T(0.1), 16)
    uvar = fill(T(0.9), 16)
    # Overrides:
    # X5: 0.9 <= X5 <= 1.0
    lvar[5] = T(0.9); uvar[5] = T(1.0)
    # X6: 0.0001 <= X6 <= 0.1
    lvar[6] = T(0.0001); uvar[6] = T(0.1)
    # X11: 1.0 <= X11 <= 1000.0
    lvar[11] = T(1.0); uvar[11] = T(1000.0)
    # X12: 0.000001 <= X12 <= 500.0
    lvar[12] = T(0.000001); uvar[12] = T(500.0)
    # X13: 1.0 <= X13 <= 500.0
    lvar[13] = T(1.0); uvar[13] = T(500.0)
    # X14: 500.0 <= X14 <= 1000.0
    lvar[14] = T(500.0); uvar[14] = T(1000.0)
    # X15: 500.0 <= X15 <= 1000.0
    lvar[15] = T(500.0); uvar[15] = T(1000.0)
    # X16: 0.000001 <= X16 <= 500.0
    lvar[16] = T(0.000001); uvar[16] = T(500.0)

    start_vals = T[0.8, 0.83, 0.85, 0.87, 0.90, 0.10, 0.12, 0.19,
                   0.25, 0.29, 512.0, 13.1, 71.8, 640.0, 650.0, 5.7]

    ec = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(ec, x, 16; lvar = lvar, uvar = uvar, start = start_vals)

    # Element type functions (inline in expressions):
    # INV(X)         = 1/X
    # 2PR(X,Y)       = X*Y
    # QT(X,Y)        = X/Y
    # QTQT(X,Y,W,Z)  = X*W/(Y*Z)
    # 2PRRC(X,Y,Z)   = X*Y/Z
    # QTRC(X,Y,Z)    = X/(Y*Z)
    # SQQT(X,Y)      = X^2/Y

    # Objective: N OBJ (identity type, no SCALE)
    # Linear: X12*1.262626 + X13*1.262626 + X14*1.262626 + X15*1.262626 + X16*1.262626
    # Elements (from GROUP USES E OBJ): E1..-1.231060 + E2..-1.231060 + ... + E5..-1.231060
    # where E1=2PR(x[1],x[12]), E2=2PR(x[2],x[13]), E3=2PR(x[3],x[14]),
    #       E4=2PR(x[4],x[15]), E5=2PR(x[5],x[16])
    # CONSTANT = 0.0 → v = linear + elements - 0
    ExaModels.@add_obj(ec,
        T(1.262626)*x[12] + T(1.262626)*x[13] + T(1.262626)*x[14] +
        T(1.262626)*x[15] + T(1.262626)*x[16] +
        T(-1.231060)*x[1]*x[12] + T(-1.231060)*x[2]*x[13] +
        T(-1.231060)*x[3]*x[14] + T(-1.231060)*x[4]*x[15] +
        T(-1.231060)*x[5]*x[16])

    # Constraint C0: G type (≥ 50.0), RANGE=200.0 → 50 ≤ v ≤ 250, lcon=0, ucon=200
    # v = 1.262626*(x[12]+x[13]+x[14]+x[15]+x[16])
    #     - 1.231060*(x[1]*x[12]+x[2]*x[13]+x[3]*x[14]+x[4]*x[15]+x[5]*x[16]) - 50
    ExaModels.@add_con(ec,
        T(1.262626)*x[12] + T(1.262626)*x[13] + T(1.262626)*x[14] +
        T(1.262626)*x[15] + T(1.262626)*x[16] +
        T(-1.231060)*x[1]*x[12] + T(-1.231060)*x[2]*x[13] +
        T(-1.231060)*x[3]*x[14] + T(-1.231060)*x[4]*x[15] +
        T(-1.231060)*x[5]*x[16] - T(50.0);
        lcon = T(0.0), ucon = T(200.0))

    # C1: L type (≤ 0), DEFAULT RHS=1.0
    # linear: 0.975*x[1]
    # elements: E6=QT(x[1],x[6])*0.034750, E7=SQQT(x[1],x[6])*(-0.00975)
    # v = 0.975*x[1] + 0.034750*(x[1]/x[6]) - 0.00975*(x[1]^2/x[6]) - 1.0 ≤ 0
    ExaModels.@add_con(ec,
        T(0.975)*x[1] + T(0.034750)*x[1]/x[6] - T(0.00975)*x[1]^2/x[6] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C2: L type, 0.975*x[2] + 0.034750*QT(x[2],x[7]) - 0.00975*SQQT(x[2],x[7]) - 1.0 ≤ 0
    ExaModels.@add_con(ec,
        T(0.975)*x[2] + T(0.034750)*x[2]/x[7] - T(0.00975)*x[2]^2/x[7] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C3: L type, 0.975*x[3] + 0.034750*QT(x[3],x[8]) - 0.00975*SQQT(x[3],x[8]) - 1.0 ≤ 0
    ExaModels.@add_con(ec,
        T(0.975)*x[3] + T(0.034750)*x[3]/x[8] - T(0.00975)*x[3]^2/x[8] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C4: L type, 0.975*x[4] + 0.034750*QT(x[4],x[9]) - 0.00975*SQQT(x[4],x[9]) - 1.0 ≤ 0
    ExaModels.@add_con(ec,
        T(0.975)*x[4] + T(0.034750)*x[4]/x[9] - T(0.00975)*x[4]^2/x[9] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C5: L type, 0.975*x[5] + 0.034750*QT(x[5],x[10]) - 0.00975*SQQT(x[5],x[10]) - 1.0 ≤ 0
    ExaModels.@add_con(ec,
        T(0.975)*x[5] + T(0.034750)*x[5]/x[10] - T(0.00975)*x[5]^2/x[10] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C6: L type, linear: none; elements: E16+E17-E18 = QT(x[6],x[7]) + QTQT(x[1],x[7],x[12],x[11]) - QTQT(x[6],x[7],x[12],x[11])
    # v = x[6]/x[7] + x[1]*x[12]/(x[7]*x[11]) - x[6]*x[12]/(x[7]*x[11]) - 1.0 ≤ 0
    ExaModels.@add_con(ec,
        x[6]/x[7] + x[1]*x[12]/(x[7]*x[11]) - x[6]*x[12]/(x[7]*x[11]) - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C7: L type, linear: X13*(-0.002)
    # elements: E19=QT(x[7],x[8]), E20=2PRRC(x[7],x[12],x[8])*0.002,
    #           E21=2PRRC(x[2],x[13],x[8])*0.002, E22=2PRRC(x[1],x[12],x[8])*(-0.002)
    # v = -0.002*x[13] + QT(x[7],x[8]) + 0.002*2PRRC(x[7],x[12],x[8])
    #     + 0.002*2PRRC(x[2],x[13],x[8]) - 0.002*2PRRC(x[1],x[12],x[8]) - 1.0 ≤ 0
    ExaModels.@add_con(ec,
        T(-0.002)*x[13] + x[7]/x[8] + T(0.002)*x[7]*x[12]/x[8] +
        T(0.002)*x[2]*x[13]/x[8] - T(0.002)*x[1]*x[12]/x[8] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C8: L type, linear: X8*1.0 + X9*1.0
    # elements: E23=2PR(x[8],x[13])*0.002, E24=2PR(x[3],x[14])*0.002,
    #           E25=2PR(x[2],x[13])*(-0.002), E26=2PR(x[9],x[14])*(-0.002)
    # v = x[8] + x[9] + 0.002*(x[8]*x[13]+x[3]*x[14]-x[2]*x[13]-x[9]*x[14]) - 1.0 ≤ 0
    ExaModels.@add_con(ec,
        x[8] + x[9] + T(0.002)*x[8]*x[13] + T(0.002)*x[3]*x[14] -
        T(0.002)*x[2]*x[13] - T(0.002)*x[9]*x[14] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C9: L type, linear: none
    # elements: E27=QT(x[9],x[3]), E28=QTQT(x[4],x[3],x[15],x[14]),
    #           E29=QTRC(x[10],x[3],x[14])*500.0, E30=QTRC(x[9],x[3],x[14])*(-500.0),
    #           E31=QTQT(x[8],x[3],x[15],x[14])*(-1.0)
    # v = QT(x[9],x[3]) + QTQT(x[4],x[3],x[15],x[14])
    #     + 500*QTRC(x[10],x[3],x[14]) - 500*QTRC(x[9],x[3],x[14])
    #     - QTQT(x[8],x[3],x[15],x[14]) - 1.0 ≤ 0
    ExaModels.@add_con(ec,
        x[9]/x[3] + x[4]*x[15]/(x[3]*x[14]) +
        T(500.0)*x[10]/(x[3]*x[14]) - T(500.0)*x[9]/(x[3]*x[14]) -
        x[8]*x[15]/(x[3]*x[14]) - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C10: L type, linear: none
    # elements: E32=QTQT(x[5],x[4],x[16],x[15]), E33=QT(x[10],x[4]),
    #           E34=INV(x[15])*500.0, E35=QT(x[16],x[15])*(-1.0),
    #           E36=QTRC(x[10],x[4],x[15])*(-500.0)
    # v = QTQT(x[5],x[4],x[16],x[15]) + QT(x[10],x[4]) + 500*INV(x[15])
    #     - QT(x[16],x[15]) - 500*QTRC(x[10],x[4],x[15]) - 1.0 ≤ 0
    ExaModels.@add_con(ec,
        x[5]*x[16]/(x[4]*x[15]) + x[10]/x[4] + T(500.0)/x[15] -
        x[16]/x[15] - T(500.0)*x[10]/(x[4]*x[15]) - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C11: L type, linear: X16*0.002
    # elements: E37=INV(x[4])*0.9, E38=2PRRC(x[5],x[16],x[4])*(-0.002)
    # v = 0.002*x[16] + 0.9*INV(x[4]) - 0.002*2PRRC(x[5],x[16],x[4]) - 1.0 ≤ 0
    #   = 0.002*x[16] + 0.9/x[4] - 0.002*x[5]*x[16]/x[4] - 1.0 ≤ 0
    ExaModels.@add_con(ec,
        T(0.002)*x[16] + T(0.9)/x[4] - T(0.002)*x[5]*x[16]/x[4] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C12: L type, linear: X11*0.002 + X12*(-0.002); no elements
    # v = 0.002*x[11] - 0.002*x[12] - 1.0 ≤ 0
    ExaModels.@add_con(ec, T(0.002)*x[11] - T(0.002)*x[12] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C13: L type, linear: none; elements: E39=QT(x[12],x[11])
    # v = QT(x[12],x[11]) - 1.0 = x[12]/x[11] - 1.0 ≤ 0
    ExaModels.@add_con(ec, x[12]/x[11] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C14: L type, linear: none; elements: E40=QT(x[4],x[5])
    # v = x[4]/x[5] - 1.0 ≤ 0
    ExaModels.@add_con(ec, x[4]/x[5] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C15: L type, linear: none; elements: E41=QT(x[3],x[4])
    # v = x[3]/x[4] - 1.0 ≤ 0
    ExaModels.@add_con(ec, x[3]/x[4] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C16: L type, linear: none; elements: E42=QT(x[2],x[3])
    # v = x[2]/x[3] - 1.0 ≤ 0
    ExaModels.@add_con(ec, x[2]/x[3] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C17: L type, linear: none; elements: E43=QT(x[1],x[2])
    # v = x[1]/x[2] - 1.0 ≤ 0
    ExaModels.@add_con(ec, x[1]/x[2] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C18: L type, linear: none; elements: E44=QT(x[9],x[10])
    # v = x[9]/x[10] - 1.0 ≤ 0
    ExaModels.@add_con(ec, x[9]/x[10] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    # C19: L type, linear: none; elements: E45=QT(x[8],x[9])
    # v = x[8]/x[9] - 1.0 ≤ 0
    ExaModels.@add_con(ec, x[8]/x[9] - T(1.0);
        lcon = T(-Inf), ucon = T(0.0))

    return ExaModels.ExaModel(ec; kwargs...)
end
