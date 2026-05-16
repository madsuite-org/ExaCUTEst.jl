# HYDROELL: Hydroelectric reservoir management (long term), 1009 vars, 1008 constraints
# N = total time steps (computed from schedule: SAT-NO + NDTWE = complex)
# Variables: V(T) for T=0..N (volumes), with SCALE VMAX
# Objective: maximize revenue = sum A(T) * QI(V(T), V(T-1))
#   (minimize negative = -sum A(T) * QI)
# QI element: Q = ZW + (W-V)*TS, RI = AA*(V^2+V*W+W^2)*1e-12 + BB*(V+W) + G*TS
#   (where V=V(T), W=V(T-1), ZW=Z(T-1), TS=1/DT)
#   F = Q * RI = revenue
# FSCAL = FSCALE/(RHO*G) where FSCALE=-1.36e6, RHO=1000, G=9.81
#
# Time schedule:
# DT=600s, NDTM=30, NDTD=96, NDTE=18, NDTSM=42, NDTWE=216
# N = SAT-NO + NDTWE where:
# SAT-NO = SAT-AM + NDTSM = FRI-PM + NDTN + NDTSM
# FRI-PM = FRI-AM + NDTD = THU-PM + NDTN + NDTD
# THU-PM = THU-AM + NDTD = WED-PM + NDTN + NDTD
# ... this sums to:
# N = NDTN*5 + NDTD*5 + NDTN + NDTSM + NDTWE where NDTN = NDTE+NDTM = 48
# Let me compute: NDTN=48, SAT-AM = 5*(NDTN+NDTD)+NDTN = 5*144+48=768
# SAT-NO = 768+42=810, N = 810+216=1026? But problem says 1009 variables → N=1008
# Actually: NDTE+NDTM = 18+30 = 48 = NDTN
# MON-AM=0, MON-PM=30, TUE-AM=30+96=126? No:
# MON-AM=0, TUE-AM=MON-PM+NDTN=MON-PM+(NDTE+NDTM), MON-PM=MON-AM+NDTD=0+96=96
# Wait: "I+ TUE-AM MON-PM NDTN" means TUE-AM = MON-PM + NDTN = 96 + 48 = 144
# TUE-PM = TUE-AM + NDTD = 144 + 96 = 240
# WED-AM = TUE-PM + NDTN = 240 + 48 = 288
# WED-PM = WED-AM + NDTD = 288 + 96 = 384
# THU-AM = WED-PM + NDTN = 384 + 48 = 432
# THU-PM = THU-AM + NDTD = 432 + 96 = 528
# FRI-AM = THU-PM + NDTN = 528 + 48 = 576
# FRI-PM = FRI-AM + NDTD = 576 + 96 = 672
# SAT-AM = FRI-PM + NDTN = 672 + 48 = 720
# SAT-NO = SAT-AM + NDTSM = 720 + 42 = 762
# N = SAT-NO + NDTWE = 762 + 216 = 978? But 978+1=979≠1009
# Hmm. Let me try differently:
# MON-AM = NDTM = 30 (not 0)
# MON-PM = MON-AM + NDTD = 30 + 96 = 126
# TUE-AM = MON-PM + NDTN = 126 + 48 = 174
# ... continuing:
# N = 30 + 96 + 48 + 96 + 48 + 96 + 48 + 96 + 48 + 96 + 48 + 42 + 216 = 1008
# Yes! N=1008 (so 1009 variables V(0..1008))
function hydroell_model(; T = Float64, backend = nothing, kwargs...)
    RHO = T(1000)
    G = T(9.81)
    AT = T(0.52)
    AN = T(0.45)
    QMIN = T(0)
    QMAX = T(26)
    VMIN = T(60000)
    VMAX = T(736000)
    FSCALE = T(-1.36e6)
    DT = 600
    NDTM = 30; NDTD = 96; NDTE = 18; NDTSM = 42; NDTWE = 216
    NDTN = NDTE + NDTM  # 48

    MON_AM = NDTM
    MON_PM = MON_AM + NDTD
    TUE_AM = MON_PM + NDTN
    TUE_PM = TUE_AM + NDTD
    WED_AM = TUE_PM + NDTN
    WED_PM = WED_AM + NDTD
    THU_AM = WED_PM + NDTN
    THU_PM = THU_AM + NDTD
    FRI_AM = THU_PM + NDTN
    FRI_PM = FRI_AM + NDTD
    SAT_AM = FRI_PM + NDTN
    SAT_NO = SAT_AM + NDTSM
    N = SAT_NO + NDTWE

    # Set tariff for each time step T=0..N-1
    A_vals = zeros(T, N)
    for t in 0:(MON_AM-1);      A_vals[t+1] = AN; end
    for t in MON_AM:(MON_PM-1); A_vals[t+1] = AT; end
    for t in MON_PM:(TUE_AM-1); A_vals[t+1] = AN; end
    for t in TUE_AM:(TUE_PM-1); A_vals[t+1] = AT; end
    for t in TUE_PM:(WED_AM-1); A_vals[t+1] = AN; end
    for t in WED_AM:(WED_PM-1); A_vals[t+1] = AT; end
    for t in WED_PM:(THU_AM-1); A_vals[t+1] = AN; end
    for t in THU_AM:(THU_PM-1); A_vals[t+1] = AT; end
    for t in THU_PM:(FRI_AM-1); A_vals[t+1] = AN; end
    for t in FRI_AM:(FRI_PM-1); A_vals[t+1] = AT; end
    for t in FRI_PM:(SAT_AM-1); A_vals[t+1] = AN; end
    for t in SAT_AM:(SAT_NO-1); A_vals[t+1] = AT; end
    for t in SAT_NO:N-1;        A_vals[t+1] = AN; end

    # Influx Z(T) = AVZ + ZVAR*sin(T*DT) where ZVAR=(ZMAX-ZMIN)/2, AVZ=ZMIN+ZVAR
    ZMIN = T(8); ZMAX = T(11.5)
    ZVAR = (ZMAX - ZMIN) / 2
    AVZ = ZMIN + ZVAR
    Z_vals = T[AVZ + ZVAR * sin(T(t * DT)) for t in 0:N]

    # QI element parameters
    RHOG = RHO * G
    FSCAL = FSCALE / RHOG   # objective scale factor (negative, since we maximize)
    aa = T(-3.12)
    bb = T(8.7e-6)
    gg = T(160.989)
    RDT = T(1) / T(DT)
    AA = aa * RDT / T(3)   # = aa/(3*DT) times 1e-12 factor
    BB_c = T(0.5) * bb * RDT
    # QI: A = -3.12, B = 8.7e-6, G_q = 160.989
    # VV = 1e-6*V, WW = 1e-6*W
    # AA = A*TS/3 * 1e-12 = -3.12/(3*DT) * 1e-12
    # But the SIF says: RI = AA*(VV^2+VV*WW+WW^2) + BB*(V+W) + G_q*TS
    # where AA = A*TS/3 * 1e-12 (AAA = 1e-12*AA = A*TS/3*1e-24)
    # Hmm, let me recompute from element TEMPORARIES:
    # A = -3.12, B = 8.7e-6, G = 160.989
    # VV = 1e-6 * V, WW = 1e-6 * W
    # AA = A * TS / 3 = -3.12 * RDT / 3
    # AAA = 1e-12 * AA = 1e-12 * (-3.12*RDT/3)
    # BB = 0.5 * B * TS = 0.5 * 8.7e-6 * RDT
    # RI = AA*(VV^2+VV*WW+WW^2) + BB*(V+W) + G*TS
    #    = AA*(1e-12)*(V^2+V*W+W^2) + BB*(V+W) + G*RDT
    # Q = ZW + (W-V)*TS where TS=RDT

    AA_c = aa * RDT / T(3) * T(1e-12)
    BB_c2 = T(0.5) * bb * RDT
    GTS = gg * RDT

    NP1 = N + 1
    nvar = NP1  # V(0..N) scaled by VMAX

    lvar = fill(VMIN, nvar)
    uvar = fill(VMAX, nvar)
    start = fill(VMAX, nvar)

    # Fix V(0)=V(N)=VMAX
    lvar[1] = VMAX; uvar[1] = VMAX
    lvar[NP1] = VMAX; uvar[NP1] = VMAX

    # V(T) scale in SIF: ZV V(T) VMAX → variable is divided by VMAX in SIF
    # But ExaModels doesn't support scaling variables, so we model V(T) directly
    # with bounds [VMIN, VMAX].

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    ExaModels.@add_par(c, A_par, A_vals)
    ExaModels.@add_par(c, Z_par, Z_vals)

    # Objective: FSCAL * sum_{T=1}^N A(T-1) * Q * RI (scaled by FSCAL, which is negative)
    # Q = ZW + (W-V)*RDT where V=x[T+1] (=V(T)), W=x[T] (=V(T-1)), ZW=Z(T-1)
    # RI = AA_c*(V^2+V*W+W^2) + BB_c2*(V+W) + GTS
    ExaModels.@add_obj(c,
        FSCAL * A_par[t] *
        (Z_par[t] + (x[t] - x[t + 1]) * RDT) *
        (AA_c * (x[t + 1]^2 + x[t + 1] * x[t] + x[t]^2) + BB_c2 * (x[t + 1] + x[t]) + GTS)
        for t in 1:N)

    # Debit constraints: Q(T) = ZW + (W-V)/DT, must satisfy QMIN <= Q <= QMAX
    # Q(T) = Z(T-1) + (V(T-1) - V(T)) * RDT
    # → QMIN - Z(T-1) <= (V(T-1) - V(T)) * RDT <= QMAX - Z(T-1)
    # ExaModels constraint: x[t] - x[t+1] (V(T-1)-V(T)) -> residual w.r.t. range
    # Residual = Q - QMIN = Z(T-1) + (V(T-1)-V(T))*RDT - QMIN
    ExaModels.@add_con(c,
        Z_par[t] + (x[t] - x[t + 1]) * RDT - QMIN
        for t in 1:N;
        lcon = T(0), ucon = QMAX - QMIN)

    return ExaModels.ExaModel(c; kwargs...)
end
