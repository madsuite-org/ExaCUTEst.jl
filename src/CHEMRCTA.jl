# CHEMRCTA: Tubular chemical reactor model (2N variables, 2N constraints)
# Default N=2500 (n=5000)
# Variables: T(1..N) temperature, U(1..N) concentration
# Both >= 0, T(I) >= 1e-7; start = 1.0
# REAC element: F = U*exp(G - G/T), G=GAMMA=25.0
# Constraint GU(I): concentration equation
# Constraint GT(I): temperature equation
# Computed coefficients depend on PEM=1, PEH=5, D=0.135, B=0.5, BETA=2, GAMMA=25
function chemrcta_model(N = 2500; T = Float64, backend = nothing, kwargs...)
    PEM  = T(1.0)
    PEH  = T(5.0)
    D    = T(0.135)
    B    = T(0.5)
    BETA = T(2.0)
    GAMA = T(25.0)   # GAMMA (Julia keyword)

    H = T(1) / T(N - 1)

    # Concentration coefficients
    oneH2PEM = T(1) / (H * H * PEM)
    HPEM     = PEM * H
    CU1      = T(1) - HPEM    # coeff of U(2) in GU(1): -(HPEM-1)*(-1) — check SIF
    # SIF: CU1 = -HPEM * 1.0 (RA CU1 -HPEM 1.0 → CU1 = -HPEM + 1.0? No: RM -HPEM HPEM -1 → -HPEM = -PEM*H
    # RA CU1 -HPEM 1.0 → CU1 = -HPEM + 1.0 = 1 - PEM*H
    # Actually from SIF: RM -HPEM HPEM -1.0; RA CU1 -HPEM 1.0 → CU1 = 1 - HPEM = 1 - PEM*H
    # And in GT: CT1 = 1 - HPEH (similarly)

    # For concentration:
    # CUI_m1 = 1/(H^2*PEM) + 1/H
    # CUI    = -2/(H^2*PEM) - 1/H  (the -D term comes from element separately)
    # CUI_p1 = 1/(H^2*PEM)  (same as 1/H2PEM)

    one_H2PEM = T(1) / (H * H * PEM)
    neg_H2PM  = -T(1) / (H * H * PEM)
    neg2_H2PM = T(-2) / (H * H * PEM)
    neg1_H    = T(-1) / H
    pos1_H    = T(1) / H

    CUI_m1 = one_H2PEM + pos1_H
    CUI    = neg2_H2PM + neg1_H
    CUI_p1 = one_H2PEM
    CU1_val = T(1) - PEM * H   # coefficient for U(2) in boundary eqn GU(1)

    # For temperature:
    one_H2PEH = T(1) / (H * H * PEH)
    neg_H2PH  = -T(1) / (H * H * PEH)
    neg2_H2PH = T(-2) / (H * H * PEH)
    HPEH_val  = PEH * H

    CTI_m1 = one_H2PEH + pos1_H
    CTI_0  = neg2_H2PH + neg1_H - BETA   # total coeff for T(I) in GT(I)
    CTI_p1 = one_H2PEH
    CT1_val = T(1) - PEH * H

    # Boundary constants: GU(1) rhs = -HPEM (from ZE CHEMRCTA GU(1) -HPEM)
    # Wait: CONSTANTS Z CHEMRCTA GU(1) -HPEM means rhs = -HPEM = -(PEM*H)
    # Actually SIF CONSTANTS stores the value of -HPEM which is -PEM*H
    # constraint GU(1): -U(1) + CU1*U(2) + elem - rhs = 0
    # rhs for GU(1) = -HPEM = -(PEM*H)
    rhs_GU1 = -(PEM * H)
    rhs_GT1 = -(PEH * H)
    neg_D = -D
    BD    = B * D

    # Variable indexing: T(I) = I, U(I) = N + I
    t_idx(i) = i
    u_idx(i) = N + i

    lvar = fill(T(0), 2 * N)
    uvar = fill(T(Inf), 2 * N)
    for i in 1:N
        lvar[t_idx(i)] = T(1e-7)  # T(I) >= 1e-7
    end
    start = ones(T, 2 * N)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2 * N; lvar = lvar, uvar = uvar, start = start)

    # GU(1): -U(1) + CU1*U(2) - rhs = 0  (no element at boundary)
    ExaModels.@add_con(c, -x[u_idx(1)] + CU1_val * x[u_idx(2)] - rhs_GU1)

    # GU(I) for I=2..N-1: CUI_m1*U(I-1) + CUI*U(I) + CUI_p1*U(I+1) + neg_D*REAC(U(I),T(I)) = 0
    inner_con = [(ui = u_idx(i), tim1 = u_idx(i - 1), tip1 = u_idx(i + 1), ti = t_idx(i)) for i in 2:N-1]
    ExaModels.@add_con(c,
        CUI_m1 * x[d.tim1] + CUI * x[d.ui] + CUI_p1 * x[d.tip1] +
        neg_D * x[d.ui] * exp(GAMA - GAMA / x[d.ti])
        for d in inner_con)

    # GU(N): -U(N-1) + U(N) = 0
    ExaModels.@add_con(c, -x[u_idx(N - 1)] + x[u_idx(N)])

    # GT(1): -T(1) + CT1*T(2) - rhs = 0
    ExaModels.@add_con(c, -x[t_idx(1)] + CT1_val * x[t_idx(2)] - rhs_GT1)

    # GT(I) for I=2..N-1: BETA*T(I) + CTI_m1*T(I-1) + CTI_0*T(I) + CTI_p1*T(I+1) + BD*REAC = 0
    # Note: CTI_0 already includes -BETA, plus there's an extra BETA from "ZE GT(I) T(I) BETA"
    # Let me re-check the SIF:
    # ZE GT(I) T(I) BETA  → adds BETA to T(I) coeff
    # ZE GT(I) T(I-1) CTI-1
    # ZE GT(I) T(I) CTI   (where CTI = neg2_H2PH + neg1_H - BETA, so total T(I) coeff = BETA + CTI)
    # Wait: the SIF has TWO lines for T(I): ZE GT(I) T(I) BETA and ZE GT(I) T(I) CTI
    # These ADD to the same variable, so total T(I) coeff = BETA + CTI = BETA + (neg2_H2PH + neg1_H - BETA)
    # = neg2_H2PH + neg1_H = -2/H^2PEH - 1/H
    total_CTI = neg2_H2PH + neg1_H   # = BETA + CTI_0

    t_inner = [(ti = t_idx(i), tim1 = t_idx(i - 1), tip1 = t_idx(i + 1),
                ui = u_idx(i)) for i in 2:N-1]
    ExaModels.@add_con(c,
        CTI_m1 * x[d.tim1] + total_CTI * x[d.ti] + CTI_p1 * x[d.tip1] +
        BD * x[d.ui] * exp(GAMA - GAMA / x[d.ti])
        for d in t_inner)

    # GT(N): -T(N-1) + T(N) = 0
    ExaModels.@add_con(c, -x[t_idx(N - 1)] + x[t_idx(N)])

    return ExaModels.ExaModel(c; kwargs...)
end
