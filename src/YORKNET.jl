# YORKNET: Yorkshire water distribution network optimal control
# 312 variables, 256 equality constraints
# SIF: SOR2-AY-312-256. Source: B. Ulanicki, De Montfort University.
#
# Variable layout per time step I=1..8 (NSTEP=8):
#   q(J,I)  J=1..15: pipe/valve/pump flows      → global (I-1)*39 + J
#   h(J,I)  J=1..13: node heads                 → global (I-1)*39 + 15 + J
#   qr(J,I) J=1..4:  reservoir flows            → global (I-1)*39 + 28 + J
#   qs(J,I) J=1..2:  treatment works flows      → global (I-1)*39 + 32 + J
#   u(J,I)  J=11..15: pump control factors      → global (I-1)*39 + 34 + (J-10)
#
# Objective: sum treatment costs + pump power costs (steps 2-4 only)
# Constraints (in order): ND(J,I), EL(J,I), RD(J,I)
function yorknet_model(; T = Float64, backend = nothing, kwargs...)
    NSTEP = 8; NEL = 15; NND = 13
    S_RES = 1; E_RES = 4; S_TW = 1; E_TW = 2; S_PMP = 11; E_PMP = 15

    # Variable index helpers
    qidx(J, I)  = (I-1)*39 + J            # J=1..15
    hidx(J, I)  = (I-1)*39 + 15 + J       # J=1..13
    qridx(J, I) = (I-1)*39 + 28 + J       # J=1..4
    qsidx(J, I) = (I-1)*39 + 32 + J       # J=1..2
    uidx(J, I)  = (I-1)*39 + 34 + (J-10) # J=11..15

    nvar = 39 * NSTEP

    # Time step sizes (all equal 0.125)
    ST = T[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    MST = T[-0.125, -0.125, -0.125, -0.125, -0.125, -0.125, -0.125, -0.125]

    # Treatment works costs per time step: CTW(J,I) = BCTW(J) * ST(I)
    BCTW = T[28.34, 18.86]
    # CTW(J,I) = BCTW[J] * 0.125 (same for all I)

    # Pipe/valve MGINV constants
    MGINV = T[-3.365170e-3, -2.314284e-2, -6.631203e-3, -1.702093e-3,
              -1.205983e-2, -9.617776e-4, -1.392046e-5, -4.411625e-3,
              -2.019250e-3, -2.288437e-3]

    # Pump constants: CONA, CONB, CONC for J=11..15
    CONA = T[-0.035520, -0.07475, -0.042420, -0.040733, -0.167495]
    CONB = T[-0.054720, -9.05,    -0.005370, -0.032036, -0.0019]
    CONC = T[99.80,     110.0,    175.29,    139.6,     139.6]

    # Demand constants D(J) for J=1..13
    D = T[0.0, 0.0, -55.0, -33.0, 0.0, 0.0, 0.0, -25.0, 0.0, -17.0, 0.0, 0.0, 0.0]

    # Reservoir data
    XAR  = T[1.599, 4.6421, 30.2307, 5.3938]
    MXAR = T[-1.599, -4.6421, -30.2307, -5.3938]  # -XAR
    HGT  = T[5.77, 3.00, 131.08, 44.0]
    MXHGT = T[9.60, 7.89, 138.76, 53.34]
    STHGT = T[8.5, 6.0, 135.6, 48.5]
    MSTVOL = T[STHGT[j]*XAR[j] for j in 1:4]  # initial volumes
    # MSTVOL = [-13.5915, -27.8526, -4099.6549, -261.9441]... wait these should be positive
    # MSTVOL = STHGT * XAR (positive volumes)
    QSMX = T[500.0, 40.0]
    WS_init = T[94.67, 30.87]  # initial source flows (used as start point)

    # Bounds
    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    x0   = fill(T(0), nvar)

    for I in 1:NSTEP
        # Reservoir heads: h(J,I) for J=1..4
        for J in 1:E_RES
            lvar[hidx(J,I)] = HGT[J]
            uvar[hidx(J,I)] = MXHGT[J]
        end
        # Other node heads h(J,I) for J=5..13 >= 0
        for J in (E_RES+1):NND
            lvar[hidx(J,I)] = T(0)
        end
        # Treatment flows qs(J,I) in [0, QSMX(J)]
        for J in S_TW:E_TW
            lvar[qsidx(J,I)] = T(0)
            uvar[qsidx(J,I)] = QSMX[J]
        end
        # Pump controls u(J,I) in [0, 7]
        for J in S_PMP:E_PMP
            lvar[uidx(J,I)] = T(0)
            uvar[uidx(J,I)] = T(7)
            lvar[qidx(J,I)] = T(0)  # pump flows >= 0
        end
        # q(J,I) for J=1..10 and J=11..15 are free (except pumps above)
        for J in 1:(S_PMP-1)
            lvar[qidx(J,I)] = T(-Inf)
        end
        # qr(J,I) are free (XR = free in SIF)
        for J in S_RES:E_RES
            lvar[qridx(J,I)] = T(-Inf)
        end
    end

    # Start point
    for I in 1:NSTEP
        for J in 1:NEL
            x0[qidx(J,I)] = T(20)
        end
        for J in S_PMP:E_PMP
            x0[uidx(J,I)] = T(3)
        end
        x0[qsidx(1,I)] = T(25)
        x0[qsidx(2,I)] = T(50)
        # node heads default to 0 (h(J,I) for J=5..13 already 0)
        for J in S_RES:E_RES
            x0[hidx(J,I)] = T(0)  # XM DEFAULT=0
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = x0)

    # ── Objective ─────────────────────────────────────────────────────────────
    # Treatment cost: CTW(J,I) * qs(J,I) for J=1..2, I=1..8
    for I in 1:NSTEP
        for J in S_TW:E_TW
            ctw = T(BCTW[J] * 0.125)
            qs_idx = qsidx(J, I)
            ExaModels.@add_obj(c, ctw * x[qs_idx])
        end
    end

    # Pumping cost: 100 * PMP(J, q(J,I)) for J=11..15, I=2..4
    for I in 2:4
        let q11 = qidx(11, I), q12 = qidx(12, I), q13 = qidx(13, I),
            q14 = qidx(14, I), q15 = qidx(15, I)
            ExaModels.@add_obj(c, T(100) * (T(0.079)*x[q11]^2 - T(2.761)*x[q11] + T(35.014)))
            ExaModels.@add_obj(c, T(100) * (T(0.034)*x[q12]^2 + T(0.220)*x[q12] + T(6.685)))
            ExaModels.@add_obj(c, T(100) * (T(0.074)*x[q13]^2 + T(3.062)*x[q13] + T(50.357)))
            ExaModels.@add_obj(c, T(100) * (T(0.747)*x[q14]^2 - T(10.287)*x[q14]))
            ExaModels.@add_obj(c, T(100) * (T(0.747)*x[q15]^2 - T(10.287)*x[q15]))
        end
    end

    # ── Constraints: ND(J,I) mass balance (pure linear) ─────────────────────
    # ND(1,I): qr(1,I)*(-1) + qs(1,I)*(-1) + q(13,I)*(1) - D(1) = 0
    # ND(2,I): qr(2,I)*(-1) + qs(2,I)*(-1) + q(14,I)*(1) + q(15,I)*(1) - D(2) = 0
    # ND(3,I): qr(3,I)*(-1) + q(5,I)*(-1) + q(6,I)*(1) - D(3) = 0
    # ND(4,I): qr(4,I)*(-1) + q(12,I)*(1) + q(8,I)*(-1) + q(9,I)*(-1) - D(4) = 0
    # ND(5,I): q(13,I)*(-1) + q(1,I)*(1) - D(5) = 0
    # ND(6,I): q(2,I)*(1) + q(3,I)*(1) + q(12,I)*(-1) - D(6) = 0
    # ND(7,I): q(2,I)*(-1) + q(7,I)*(-1) + q(11,I)*(1) - D(7) = 0
    # ND(8,I): q(3,I)*(-1) + q(7,I)*(1) - D(8) = 0
    # ND(9,I): q(4,I)*(1) + q(5,I)*(1) + q(11,I)*(-1) - D(9) = 0
    # ND(10,I): q(4,I)*(-1) + q(6,I)*(-1) - D(10) = 0
    # ND(11,I): q(10,I)*(1) + q(14,I)*(-1) + q(15,I)*(-1) - D(11) = 0
    # ND(12,I): q(8,I)*(1) + q(10,I)*(-1) - D(12) = 0
    # ND(13,I): q(9,I)*(1) + q(1,I)*(-1) - D(13) = 0
    for I in 1:NSTEP
        ExaModels.@add_con(c, -x[qridx(1,I)] - x[qsidx(1,I)] + x[qidx(13,I)] - D[1])
        ExaModels.@add_con(c, -x[qridx(2,I)] - x[qsidx(2,I)] + x[qidx(14,I)] + x[qidx(15,I)] - D[2])
        ExaModels.@add_con(c, -x[qridx(3,I)] - x[qidx(5,I)] + x[qidx(6,I)] - D[3])
        ExaModels.@add_con(c, -x[qridx(4,I)] + x[qidx(12,I)] - x[qidx(8,I)] - x[qidx(9,I)] - D[4])
        ExaModels.@add_con(c, -x[qidx(13,I)] + x[qidx(1,I)] - D[5])
        ExaModels.@add_con(c, x[qidx(2,I)] + x[qidx(3,I)] - x[qidx(12,I)] - D[6])
        ExaModels.@add_con(c, -x[qidx(2,I)] - x[qidx(7,I)] + x[qidx(11,I)] - D[7])
        ExaModels.@add_con(c, -x[qidx(3,I)] + x[qidx(7,I)] - D[8])
        ExaModels.@add_con(c, x[qidx(4,I)] + x[qidx(5,I)] - x[qidx(11,I)] - D[9])
        ExaModels.@add_con(c, -x[qidx(4,I)] - x[qidx(6,I)] - D[10])
        ExaModels.@add_con(c, x[qidx(10,I)] - x[qidx(14,I)] - x[qidx(15,I)] - D[11])
        ExaModels.@add_con(c, x[qidx(8,I)] - x[qidx(10,I)] - D[12])
        ExaModels.@add_con(c, x[qidx(9,I)] - x[qidx(1,I)] - D[13])
    end

    # ── Constraints: EL(J,I) head-drop relations ──────────────────────────────
    # Pipes J=1..8: head_diff + MGINV(J) * sgn(q)*|q|^1.852 = 0
    # Valves J=9..10: head_diff + MGINV(J) * sgn(q)*|q|^1.852 = 0
    # Pumps J=11..15: CONA*q^2 + CONB*q*u + CONC*u^2 + (h1-h2)*u^2 = 0

    # Head differences in GROUPS section:
    # EL(1,I): h(5,I)-h(13,I); EL(2,I): h(6,I)-h(7,I); EL(3,I): h(6,I)-h(8,I)
    # EL(4,I): h(9,I)-h(10,I); EL(5,I): h(9,I)-h(3,I); EL(6,I): h(3,I)-h(10,I)
    # EL(7,I): h(8,I)-h(7,I); EL(8,I): h(12,I)-h(4,I)
    # EL(9,I): h(13,I)-h(4,I); EL(10,I): h(11,I)-h(12,I)
    # EL(J,I) for J=11..15: no linear part; pump nonlinear terms only.
    # Pump node pairs (head1, head2) from ELEMENT USES:
    # EH(11,I): X=h(7,I), Y=h(9,I) → h(7,I)-h(9,I)
    # EH(12,I): X=h(4,I), Y=h(6,I) → h(4,I)-h(6,I)
    # EH(13,I): X=h(1,I), Y=h(5,I) → h(1,I)-h(5,I)
    # EH(14,I): X=h(2,I), Y=h(11,I) → h(2,I)-h(11,I)
    # EH(15,I): X=h(2,I), Y=h(11,I) → h(2,I)-h(11,I)
    pipe_hdiff = [
        (5, 13),  # EL(1,I)
        (6,  7),  # EL(2,I)
        (6,  8),  # EL(3,I)
        (9, 10),  # EL(4,I)
        (9,  3),  # EL(5,I)
        (3, 10),  # EL(6,I)
        (8,  7),  # EL(7,I)
        (12, 4),  # EL(8,I)
        (13, 4),  # EL(9,I)  valve
        (11,12),  # EL(10,I) valve
    ]
    pump_hdiff = [(7,9), (4,6), (1,5), (2,11), (2,11)]  # J=11..15

    for I in 1:NSTEP
        # Pipes and valves J=1..10
        # XPOW element: F(q) = sign(q)*|q|^1.852 = q*|q|^0.852
        for (j, (n1, n2)) in enumerate(pipe_hdiff)
            let q_v = qidx(j, I), h1 = hidx(n1, I), h2 = hidx(n2, I), mg = MGINV[j]
                ExaModels.@add_con(c,
                    x[h1] - x[h2] + mg * x[q_v] * abs(x[q_v])^T(0.852))
            end
        end
        # Pumps J=11..15
        for (jj, (n1, n2)) in enumerate(pump_hdiff)
            let j = jj + 10,
                q_v = qidx(j, I), u_v = uidx(j, I),
                h1 = hidx(n1, I), h2 = hidx(n2, I),
                ca = CONA[jj], cb = CONB[jj], cc = CONC[jj]
                ExaModels.@add_con(c,
                    ca * x[q_v]^2 + cb * x[q_v] * x[u_v] +
                    (cc + x[h1] - x[h2]) * x[u_v]^2)
            end
        end
    end

    # ── Constraints: RD(J,I) reservoir dynamics ───────────────────────────────
    # Step I=1: MXAR(J)*h(J,1) + MST(1)*qr(J,1) - MSTVOL(J) = 0
    for J in S_RES:E_RES
        ExaModels.@add_con(c,
            MXAR[J] * x[hidx(J,1)] + MST[1] * x[qridx(J,1)] - MSTVOL[J])
    end
    # Steps I=2..8: MXAR(J)*h(J,I) + XAR(J)*h(J,I-1) + MST(I)*qr(J,I) = 0
    for I in 2:NSTEP
        for J in S_RES:E_RES
            ExaModels.@add_con(c,
                MXAR[J] * x[hidx(J,I)] + XAR[J] * x[hidx(J,I-1)] + MST[I] * x[qridx(J,I)])
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end
