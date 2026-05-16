# AVION2: Dassault France avion (airplane design) problem
# 49 variables, 15 equality constraints
# Source: A.R. Conn, June 1993.
# SIF: OLR2-RN-49-15
#
# CUTEst variable order (determined empirically from lvar/uvar/x0):
#  x[1]=SR,  x[2]=NM,      x[3]=PK,       x[4]=EF,      x[5]=SX,
#  x[6]=LX,  x[7]=SD,      x[8]=SK,       x[9]=ST,      x[10]=IMPVOIL,
#  x[11]=LF, x[12]=AM,     x[13]=CA,      x[14]=CB,     x[15]=SO,
#  x[16]=SS, x[17]=IMPDER, x[18]=IMPK,    x[19]=DS,     x[20]=QI,
#  x[21]=PT, x[22]=MV,     x[23]=MC,      x[24]=MD,     x[25]=PD,
#  x[26]=DE, x[27]=VS,     x[28]=CR,      x[29]=PM,     x[30]=DV,
#  x[31]=MZ, x[32]=VN,     x[33]=QV,      x[34]=QF,     x[35]=EL,
#  x[36]=IMPMOT, x[37]=MS, x[38]=IMPPET,  x[39]=IMPPIL, x[40]=IMPCAN,
#  x[41]=IMPSNA, x[42]=IMPNMOT, x[43]=IMPTRAIN, x[44]=NS, x[45]=IMPFUS,
#  x[46]=SF, x[47]=LR,     x[48]=NP,      x[49]=NG
#
# Element types (from ELEMENTS section):
#  2PR(Y,Z)       = Y*Z
#  QD/SQ(W,X,Y,Z) = (W - X - Y*Z) / Z^2
#  12(Y,Z)        = Y/Z
#  12/1(X,Y,Z)    = X*Y^2/Z
#  SQ(Z)          = Z^2
#  QT(Y,Z)        = Y/Z
#  1/LIN(Y,Z)     = Y/(Z+20)
#  SQRT(Z)        = sqrt(Z)
#  SURD(X,Y,Z)    = X^(3/2)*Y/sqrt(Z)
#  SQPRD(Y,Z)     = Y^2*Z
#  CB/SQQD(W,X,Y,Z) = (W - X*Y^3) / (Y^2*(3 - Z*Y))
#  SREL(V,W,X,Y,Z) = V*(15+0.15*V)*((W*X/(50*Y*Z))^(3/2) + 8)
function avion2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    # Variable bounds and starting point (CUTEst ordering)
    # SIF START POINT uses label "AVION" (not "AVION2"), so CUTEst uses those XV values.
    ExaModels.@add_var(c, x, 49;
        lvar = T[10.0, 1.0, 0.0, 0.0, 7.0, 1.5, 2.0, 2.0, 30.0, 500.0,
                 0.001, 0.0, -0.2, 0.1, 0.0, 0.0, 100.0, 500.0, 0.0, 1000.0,
                 2.0, 2000.0, 3000.0, 5000.0, 0.2, 0.0, 0.0, 100.0, 4.0, 0.0,
                 500.0, 10.0, 250.0, 750.0, 2.0, 10.0, 100.0, 100.0, 200.0, 120.0,
                 700.0, 35.0, 250.0, 1.0, 500.0, 20.0, 0.0, 1.0, 1.0],
        uvar = T[150.0, 2.0, 10.0, 5.0, 120.0, 8.0, 20.0, 30.0, 500.0, 5000.0,
                 20.0, 10.0, -0.001, 2.0, 1.0, 2.0, 1000.0, 5000.0, 2.0, 20000.0,
                 30.0, 20000.0, 30000.0, 50000.0, 0.8, 1.0, 20.0, 400.0, 15.0, 10.0,
                 10000.0, 50.0, 5000.0, 15000.0, 20.0, 5000.0, 1000.0, 3000.0, 400.0, 240.0,
                 1900.0, 70.0, 3000.0, 5.0, 5000.0, 200.0, 10.0, 2.0, 2.0],
        start = T[27.452, 1.0, 10.0, 0.0, 19.217, 1.5, 3.5688, 4.0696, 34.315, 5000.0,
                  5.1306, 0.0, -0.14809, 0.7598, 0.0, 0.0, 114.7, 500.0, 0.0, 2325.6,
                  5.6788, 14197.0, 12589.0, 28394.0, 0.2, 0.0, 0.0, 100.0, 15.0, 0.0,
                  500.0, 10.0, 814.9, 3140.5, 4.9367, 190.85, 1000.0, 100.0, 200.0, 120.0,
                  700.0, 35.0, 1945.0, 1.0, 1760.5, 88.025, 1.5, 1.0, 1.0])

    # Using SIF variable names in comments; Julia indices per CUTEst ordering above.
    # SR=x[1], NM=x[2], PK=x[3], EF=x[4], SX=x[5], LX=x[6], SD=x[7], SK=x[8]
    # ST=x[9], IMPVOIL=x[10], LF=x[11], AM=x[12], CA=x[13], CB=x[14], SO=x[15], SS=x[16]
    # IMPDER=x[17], IMPK=x[18], DS=x[19], QI=x[20], PT=x[21], MV=x[22], MC=x[23], MD=x[24]
    # PD=x[25], DE=x[26], VS=x[27], CR=x[28], PM=x[29], DV=x[30], MZ=x[31], VN=x[32]
    # QV=x[33], QF=x[34], EL=x[35], IMPMOT=x[36], MS=x[37], IMPPET=x[38], IMPPIL=x[39]
    # IMPCAN=x[40], IMPSNA=x[41], IMPNMOT=x[42], IMPTRAIN=x[43], NS=x[44], IMPFUS=x[45]
    # SF=x[46], LR=x[47], NP=x[48], NG=x[49]

    # ── Objective (SQUARE groups: contribution = v^2) ─────────────────────────
    # E4: XN SK; EL1 = 2PR(PK,SR)*(-0.01) → v = SK - 0.01*PK*SR
    ExaModels.@add_obj(c, (x[8] - T(0.01)*x[3]*x[1])^2)

    # E6: XN CA; EL2 = QD/SQ(SS,SO,CB,LF)*(-1) → QD/SQ = (SS-SO-CB*LF)/LF^2
    #   v = CA - (SS - SO - CB*LF)/LF^2
    ExaModels.@add_obj(c, (x[13] - (x[16] - x[15] - x[14]*x[11])/x[11]^2)^2)

    # E7: XN AM -2, SO 1, SS 1; EL3 = 12(EF,LF)*0.01 → 12 = EF/LF
    #   v = -2*AM + SO + SS + 0.01*EF/LF
    ExaModels.@add_obj(c, (-T(2)*x[12] + x[15] + x[16] + T(0.01)*x[4]/x[11])^2)

    # E8: XN AM 1; EL4 = 12/1(SO,CB,CA)*(-0.25) → 12/1 = SO*CB^2/CA
    #   v = AM - 0.25*SO*CB^2/CA
    ExaModels.@add_obj(c, (x[12] - T(0.25)*x[15]*x[14]^2/x[13])^2)

    # E9: XN IMPDER 1, SD -27.5; EL5 = SQ(SD)*(-1.3) → SQ = SD^2
    #   v = IMPDER - 27.5*SD - 1.3*SD^2
    ExaModels.@add_obj(c, (x[17] - T(27.5)*x[7] - T(1.3)*x[7]^2)^2)

    # E10: XN IMPK 1, SK -70; EL6 = SQ(SK)*8.6 → SQ = SK^2
    #   v = IMPK - 70*SK + 8.6*SK^2
    ExaModels.@add_obj(c, (x[18] - T(70.0)*x[8] + T(8.6)*x[8]^2)^2)

    # E13: XN QI 1 (constant 1000 → group rhs=1000);
    #   EL7 = SQ(MV)*(1/24000) → SQ = MV^2
    #   v = QI + MV^2/24000 - 1000  [SIF CONSTANT E13 = 1000 → residual = v - 1000]
    #   But E13 is XN (objective, N type), so rhs just shifts: obj contributes (v-1000)^2?
    #   Wait: for objective groups CONSTANTS shifts the linear part, so v = QI + MV^2/24000 - 1000
    #   Actually in SIF, CONSTANTS shifts the group RHS for the group value computation:
    #   v_g = linear_part + elem_part - rhs   where rhs = CONSTANT = 1000
    #   So v_E13 = QI + MV^2/24000 - 1000
    ExaModels.@add_obj(c, (x[20] + x[22]^2/T(24000.0) - T(1000.0))^2)

    # E14: XN PT 1000; EL8 = 2PR(MD,PD)*(-1) → 2PR = MD*PD
    #   v = 1000*PT - MD*PD
    ExaModels.@add_obj(c, (T(1000.0)*x[21] - x[24]*x[25])^2)

    # E16: XN VN 1, VS 1, QF (1/790); CONSTANT E16 = -2;
    #   EL9 = QT(MZ,CR)*(-1) → QT = MZ/CR; EL10 = 2PR(DV,PT)*(1) → DV*PT
    #   v = VN + VS + QF/790 - MZ/CR + DV*PT - (-2) = VN + VS + QF/790 - MZ/CR + DV*PT + 2
    ExaModels.@add_obj(c, (x[32] + x[27] + x[34]/T(790.0) - x[31]/x[28] + x[30]*x[21] + T(2.0))^2)

    # E18: XN IMPMOT 1; EL11 = 1/LIN(PT,PM)*(-1000) → PT/(PM+20);
    #   EL12 = SQRT(PT)*(-12) → sqrt(PT)
    #   v = IMPMOT - 1000*PT/(PM+20) - 12*sqrt(PT)
    ExaModels.@add_obj(c, (x[36] - T(1000.0)*x[21]/(x[29] + T(20.0)) - T(12.0)*sqrt(x[21]))^2)

    # E26: XN ST 1; EL13 = 2PR(SR,NM)*(-1.25) → SR*NM
    #   v = ST - 1.25*SR*NM
    ExaModels.@add_obj(c, (x[9] - T(1.25)*x[1]*x[2])^2)

    # E27: XN SR 1; EL14 = QT(MD,MS)*(-1) → MD/MS
    #   v = SR - MD/MS
    ExaModels.@add_obj(c, (x[1] - x[24]/x[37])^2)

    # E28: XN QV 1; EL15 = SURD(SX,EL,LX)*(-2.4) → SX^(3/2)*EL/sqrt(LX)
    #   v = QV - 2.4*SX^(3/2)*EL/sqrt(LX)
    ExaModels.@add_obj(c, (x[33] - T(2.4)*x[5]^(T(3)/T(2))*x[35]/sqrt(x[6]))^2)

    # E29: XN SO 1; EL16 = SQPRD(DE,PT)*(-0.785) → DE^2*PT
    #   v = SO - 0.785*DE^2*PT
    ExaModels.@add_obj(c, (x[15] - T(0.785)*x[26]^2*x[21])^2)

    # E30: XN SS 1; EL17 = SQPRD(DS,PT)*(-0.785) → DS^2*PT
    #   v = SS - 0.785*DS^2*PT
    ExaModels.@add_obj(c, (x[16] - T(0.785)*x[19]^2*x[21])^2)

    # E31: XN CB 1; EL18 = CB/SQQD(VN,CA,LF,SO)*(-2) → (VN-CA*LF^3)/(LF^2*(3-SO*LF))
    #   v = CB - 2*(VN - CA*LF^3)/(LF^2*(3 - SO*LF))
    ExaModels.@add_obj(c, (x[14] - T(2.0)*(x[32] - x[13]*x[11]^3)/(x[11]^2*(T(3.0) - x[15]*x[11])))^2)

    # E32: XN IMPVOIL 1; EL19 = SREL(SX,MC,LX,SR,EL)*(-1.15)
    #   SREL = SX*(15+0.15*SX)*((MC*LX/(50*SR*EL))^(3/2) + 8)
    #   v = IMPVOIL - 1.15*SX*(15+0.15*SX)*((MC*LX/(50*SR*EL))^(3/2) + 8)
    ExaModels.@add_obj(c, (x[10] - T(1.15)*x[5]*(T(15.0) + T(0.15)*x[5])*((x[23]*x[6]/(T(50.0)*x[1]*x[35]))^(T(3)/T(2)) + T(8.0)))^2)

    # ── Equality constraints (XE groups, lcon=ucon=0) ────────────────────────
    # CUTEst may reorder constraints (linear-only groups first).
    # Verify empirically: check which formula gives near-zero at x0.
    # E1:  SD - 0.13*SR = 0
    ExaModels.@add_con(c, x[7] - T(0.13)*x[1])
    # E2:  SX - 0.7*SR = 0
    ExaModels.@add_con(c, x[5] - T(0.7)*x[1])
    # E3:  LX - LR = 0
    ExaModels.@add_con(c, x[6] - x[47])
    # E5:  SF - ST - 2*SD - 2*SX - 2*SK = 0
    ExaModels.@add_con(c, x[46] - x[9] - T(2)*x[7] - T(2)*x[5] - T(2)*x[8])
    # E11: IMPFUS - 20*SF = 0
    ExaModels.@add_con(c, x[45] - T(20.0)*x[46])
    # E12: MD - 2*MV = 0
    ExaModels.@add_con(c, x[24] - T(2.0)*x[22])
    # E15: QF - QI - QV = 0
    ExaModels.@add_con(c, x[34] - x[20] - x[33])
    # E17: IMPTRAIN - 0.137*MV = 0
    ExaModels.@add_con(c, x[43] - T(0.137)*x[22])
    # E19: IMPNMOT - 35*NM = 0
    ExaModels.@add_con(c, x[42] - T(35.0)*x[2])
    # E20: IMPPET - 0.043*QI = 0
    ExaModels.@add_con(c, x[38] - T(0.043)*x[20])
    # E21: IMPPIL - 200*NP = 0
    ExaModels.@add_con(c, x[39] - T(200.0)*x[48])
    # E22: IMPCAN - 120*NG = 0
    ExaModels.@add_con(c, x[40] - T(120.0)*x[49])
    # E23: IMPSNA - 300*NS - 400 = 0  (CONSTANT 400)
    ExaModels.@add_con(c, x[41] - T(300.0)*x[44] - T(400.0))
    # E24: MC - MV + 95*NP + 70*NG + 660*NM + 0.5*QI - 380 = 0  (CONSTANT 380)
    ExaModels.@add_con(c, x[23] - x[22] + T(95.0)*x[48] + T(70.0)*x[49] + T(660.0)*x[2] + T(0.5)*x[20] - T(380.0))
    # E25: MZ - IMPTRAIN + IMPNMOT + IMPPET + IMPPIL + IMPCAN + IMPSNA + 290 = 0  (CONSTANT -290)
    ExaModels.@add_con(c, x[31] - x[43] + x[42] + x[38] + x[39] + x[40] + x[41] + T(290.0))

    return ExaModels.ExaModel(c; kwargs...)
end
