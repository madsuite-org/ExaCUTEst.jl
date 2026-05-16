# HYDROELS: Hydroelectric reservoir management (short term), 169 vars, 168 constraints
# Same structure as HYDROELL but with different time parameters:
# DT=3600s, NDTM=5, NDTD=16, NDTE=3, NDTSM=7, NDTWE=36 → N=168
function hydroels_model(; T = Float64, backend = nothing, kwargs...)
    RHO = T(1000)
    G = T(9.81)
    AT = T(0.52)
    AN = T(0.45)
    QMIN = T(0)
    QMAX = T(26)
    VMIN = T(60000)
    VMAX = T(736000)
    FSCALE = T(-1.36e6)
    DT = 3600
    NDTM = 5; NDTD = 16; NDTE = 3; NDTSM = 7; NDTWE = 36
    NDTN = NDTE + NDTM  # 8

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

    # Influx Z(T) = AVZ + ZVAR*sin(T*DT)
    ZMIN = T(8); ZMAX = T(11.5)
    ZVAR = (ZMAX - ZMIN) / 2
    AVZ = ZMIN + ZVAR
    Z_vals = T[AVZ + ZVAR * sin(T(t * DT)) for t in 0:N]

    RHOG = RHO * G
    FSCAL = FSCALE / RHOG
    aa = T(-3.12)
    bb = T(8.7e-6)
    gg = T(160.989)
    RDT = T(1) / T(DT)
    AA_c = aa * RDT / T(3) * T(1e-12)
    BB_c2 = T(0.5) * bb * RDT
    GTS = gg * RDT

    NP1 = N + 1
    nvar = NP1

    lvar = fill(VMIN, nvar)
    uvar = fill(VMAX, nvar)
    start = fill(VMAX, nvar)

    # Fix V(0)=V(N)=VMAX
    lvar[1] = VMAX; uvar[1] = VMAX
    lvar[NP1] = VMAX; uvar[NP1] = VMAX

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    ExaModels.@add_par(c, A_par, A_vals)
    ExaModels.@add_par(c, Z_par, Z_vals)

    ExaModels.@add_obj(c,
        FSCAL * A_par[t] *
        (Z_par[t] + (x[t] - x[t + 1]) * RDT) *
        (AA_c * (x[t + 1]^2 + x[t + 1] * x[t] + x[t]^2) + BB_c2 * (x[t + 1] + x[t]) + GTS)
        for t in 1:N)

    ExaModels.@add_con(c,
        Z_par[t] + (x[t] - x[t + 1]) * RDT - QMIN
        for t in 1:N;
        lcon = T(0), ucon = QMAX - QMIN)

    return ExaModels.ExaModel(c; kwargs...)
end
