# ANTWERP: Synthetic population estimation for Belgian municipality (Antwerp)
# 27 variables: 5+3+3 child-probabilities, 3+3+3 adult-probabilities, 3 household counts,
#               2 age-class children counts, 2 age-class adult counts
# Objective: nonlinear least-squares (L2 groups) over household size predictions
# Constraints: 8 equality (probability/age-class) + 2 inequality
# Source: Ph. Toint, Apr 2006
function antwerp_model(; T = Float64, backend = nothing, kwargs...)
    # Variable indices
    # P1F=1,P2F=2,P3F=3,P4F=4,P5F=5
    # P1W=6,P2W=7,P3W=8
    # P1M=9,P2M=10,P3M=11
    # Q0F=12,Q1F=13,Q2F=14
    # Q0W=15,Q1W=16,Q2W=17
    # Q0M=18,Q1M=19,Q2M=20
    # NF=21,NW=22,NM=23
    # NC2=24,NA2=25,NC3=26,NA3=27
    P1F=1;P2F=2;P3F=3;P4F=4;P5F=5
    P1W=6;P2W=7;P3W=8
    P1M=9;P2M=10;P3M=11
    Q0F=12;Q1F=13;Q2F=14
    Q0W=15;Q1W=16;Q2W=17
    Q0M=18;Q1M=19;Q2M=20
    NF_=21;NW_=22;NM_=23
    NC2=24;NA2=25;NC3=26;NA3=27

    # Problem data
    M3 = T(23844); M4 = T(16323); M5 = T(6613); M6 = T(2535); M7 = T(1109); M8 = T(1667)
    M1F_ = T(16405); M2F_ = T(13647); M3F_ = T(9895)
    M1M_ = T(4041);  M2M_ = T(1634);  M3M_ = T(637)
    M1W_ = T(10966); M2W_ = T(4566);  M3W_ = T(1921)
    N0 = T(15866); N1 = T(59832); N2 = T(61929); N3 = T(32321); N4 = T(73650)
    NINF = T(180055); NINN = T(47677)

    SNF = M1F_ + M2F_ + M3F_   # 39947
    SNW = M1W_ + M2W_ + M3W_   # 17453
    SNM = M1M_ + M2M_ + M3M_   # 6312
    N01  = N0 + N1
    N23_2 = T(0.5)*(N2 + N3)

    # Starting values
    SP1F = M1F_ / SNF; SP2F = M2F_ / SNF
    SP1W = M1W_ / SNW; SP2W = M2W_ / SNW; SP3W = M3W_ / SNW
    SP1M = M1M_ / SNM; SP2M = M2M_ / SNM; SP3M = M3M_ / SNM

    # Bounds
    lvar = fill(T(0), 27)
    uvar = fill(T(1), 27)
    # NF, NW, NM: PL → lvar=0, uvar=Inf
    lvar[NF_] = T(0); uvar[NF_] = T(Inf)
    lvar[NW_] = T(0); uvar[NW_] = T(Inf)
    lvar[NM_] = T(0); uvar[NM_] = T(Inf)
    # NC2, NA2: 0 <= . <= N2
    lvar[NC2] = T(0); uvar[NC2] = N2
    lvar[NA2] = T(0); uvar[NA2] = N2
    # NC3, NA3: 0 <= . <= N3
    lvar[NC3] = T(0); uvar[NC3] = N3
    lvar[NA3] = T(0); uvar[NA3] = N3

    start = zeros(T, 27)
    start[P1F]=SP1F; start[P2F]=SP2F; start[P3F]=T(0.15); start[P4F]=T(0.10); start[P5F]=T(0.05)
    start[P1W]=SP1W; start[P2W]=SP2W; start[P3W]=SP3W
    start[P1M]=SP1M; start[P2M]=SP2M; start[P3M]=SP3M
    start[Q0F]=T(0.6); start[Q1F]=T(0.3); start[Q2F]=T(0.1)
    start[Q0W]=T(0.6); start[Q1W]=T(0.3); start[Q2W]=T(0.1)
    start[Q0M]=T(0.6); start[Q1M]=T(0.3); start[Q2M]=T(0.1)
    start[NF_]=SNF; start[NW_]=SNW; start[NM_]=SNM
    start[NC2]=T(0); start[NC3]=T(0); start[NA2]=N2; start[NA3]=N3

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 27; lvar = lvar, uvar = uvar, start = start)

    # Helper: p*q*n triple product and p*n double product
    # Objective: sum of (v - rhs)^2 / sigma for N-type L2 groups
    # HSZ3: sigma=M3, rhs=M3, v = P1F*Q0F*NF + P1W*Q1W*NW + P2W*Q0W*NW + P1M*Q1M*NM + P2M*Q0M*NM
    hsz3 = x[P1F]*x[Q0F]*x[NF_] + x[P1W]*x[Q1W]*x[NW_] + x[P2W]*x[Q0W]*x[NW_] +
           x[P1M]*x[Q1M]*x[NM_] + x[P2M]*x[Q0M]*x[NM_]
    ExaModels.@add_obj(c, (hsz3 - M3)^2 / M3)

    # HSZ4: sigma=M4, rhs=M4
    hsz4 = x[P2F]*x[Q0F]*x[NF_] + x[P1F]*x[Q1F]*x[NF_] + x[P1W]*x[Q2W]*x[NW_] +
           x[P2W]*x[Q1W]*x[NW_] + x[P3W]*x[Q0W]*x[NW_] + x[P1M]*x[Q2M]*x[NM_] +
           x[P2M]*x[Q1M]*x[NM_] + x[P3M]*x[Q0M]*x[NM_]
    ExaModels.@add_obj(c, (hsz4 - M4)^2 / M4)

    # HSZ5: sigma=M5, rhs=M5
    hsz5 = x[P3F]*x[Q0F]*x[NF_] + x[P2F]*x[Q1F]*x[NF_] + x[P1F]*x[Q2F]*x[NF_] +
           x[P2W]*x[Q2W]*x[NW_] + x[P3W]*x[Q1W]*x[NW_] + x[P2M]*x[Q2M]*x[NM_] +
           x[P3M]*x[Q1M]*x[NM_]
    ExaModels.@add_obj(c, (hsz5 - M5)^2 / M5)

    # HSZ6: sigma=M6, rhs=M6
    hsz6 = x[P4F]*x[Q0F]*x[NF_] + x[P3F]*x[Q1F]*x[NF_] + x[P2F]*x[Q2F]*x[NF_] +
           x[P3W]*x[Q2W]*x[NW_] + x[P3M]*x[Q2M]*x[NM_]
    ExaModels.@add_obj(c, (hsz6 - M6)^2 / M6)

    # HSZ7: sigma=M7, rhs=M7
    hsz7 = x[P5F]*x[Q0F]*x[NF_] + x[P4F]*x[Q1F]*x[NF_] + x[P3F]*x[Q2F]*x[NF_]
    ExaModels.@add_obj(c, (hsz7 - M7)^2 / M7)

    # HSZ8: sigma=M8, rhs=M8
    hsz8 = x[P5F]*x[Q1F]*x[NF_] + x[P4F]*x[Q2F]*x[NF_]
    ExaModels.@add_obj(c, (hsz8 - M8)^2 / M8)

    # HST1F: sigma=M1F, rhs=M1F
    ExaModels.@add_obj(c, (x[P1F]*x[NF_] - M1F_)^2 / M1F_)
    # HST2F: sigma=M2F, rhs=M2F
    ExaModels.@add_obj(c, (x[P2F]*x[NF_] - M2F_)^2 / M2F_)
    # HST3F: sigma=M3F, rhs=M3F
    ExaModels.@add_obj(c, ((x[P3F]+x[P4F]+x[P5F])*x[NF_] - M3F_)^2 / M3F_)
    # HST1W: sigma=M1W, rhs=M1W
    ExaModels.@add_obj(c, (x[P1W]*x[NW_] - M1W_)^2 / M1W_)
    # HST2W: sigma=M2W, rhs=M2W
    ExaModels.@add_obj(c, (x[P2W]*x[NW_] - M2W_)^2 / M2W_)
    # HST3W: sigma=M3W, rhs=M3W
    ExaModels.@add_obj(c, (x[P3W]*x[NW_] - M3W_)^2 / M3W_)
    # HST1M: sigma=M1M, rhs=M1M
    ExaModels.@add_obj(c, (x[P1M]*x[NM_] - M1M_)^2 / M1M_)
    # HST2M: sigma=M2M, rhs=M2M
    ExaModels.@add_obj(c, (x[P2M]*x[NM_] - M2M_)^2 / M2M_)
    # HST3M: sigma=M3M, rhs=M3M
    ExaModels.@add_obj(c, (x[P3M]*x[NM_] - M3M_)^2 / M3M_)

    # HCH: sigma=100*N0123, rhs=N01 (CONSTANTS: Z ANTWERP HCH N01)
    # XN HCH NC2 -1.0, NC3 -1.0; GROUP USES: L2 type with elements
    # WHCH = N0123 * 100.0; scale=WHCH
    # v_HCH = P1F*NF + 2*P2F*NF + 3*P3F*NF + 4*P4F*NF + 5*P5F*NF + P1M*NM + 2*P2M*NM + 3*P3M*NM
    #       + P1W*NW + 2*P2W*NW + 3*P3W*NW - NC2 - NC3
    # rhs = N01
    # WHCH = (N0+N1+N2/2+N3/2)*100 (approximation), but ZN HCH scale=WHCH
    # Group value: v = linear(-NC2-NC3) + elements(child count)
    # HCH with L2 and scale WHCH: contribution = (v - N01)^2 / WHCH
    WHCH = (N01 + N23_2) * T(100)
    hch = (x[P1F]+T(2)*x[P2F]+T(3)*x[P3F]+T(4)*x[P4F]+T(5)*x[P5F])*x[NF_] +
          (x[P1M]+T(2)*x[P2M]+T(3)*x[P3M])*x[NM_] +
          (x[P1W]+T(2)*x[P2W]+T(3)*x[P3W])*x[NW_] -
          x[NC2] - x[NC3]
    ExaModels.@add_obj(c, (hch - N01)^2 / WHCH)

    # HAD: sigma=WHAD=N234*100, rhs=N4
    # N234 = N4 + N2/2 + N3/2
    WHAD = (N4 + N23_2) * T(100)
    had = (T(2)*x[Q0F]+T(3)*x[Q1F]+T(4)*x[Q2F])*x[NF_] +
          x[Q0M]*x[NM_] + (T(2)*x[Q1M]+T(3)*x[Q2M])*x[NM_] +
          x[Q0W]*x[NW_] + (T(2)*x[Q1W]+T(3)*x[Q2W])*x[NW_] -
          x[NA2] - x[NA3]
    ExaModels.@add_obj(c, (had - N4)^2 / WHAD)

    # HINF: sigma=?, rhs=NINF, ZN HINF (no scale given explicitly? let's check)
    # ZN HINF → no scale token, default scale=1
    hinf = (x[P1F]+T(2)*x[P2F]+T(3)*x[P3F]+T(4)*x[P4F]+T(5)*x[P5F])*x[NF_] +
           (T(2)*x[Q0F]+T(3)*x[Q1F]+T(4)*x[Q2F])*x[NF_]
    ExaModels.@add_obj(c, (hinf - NINF)^2 / T(1))

    # HINN: sigma=1, rhs=NINN
    hinn = (x[P1W]+T(2)*x[P2W]+T(3)*x[P3W])*x[NW_] +
           (x[Q0W]+T(2)*x[Q1W]+T(3)*x[Q2W])*x[NW_] +
           (x[P1M]+T(2)*x[P2M]+T(3)*x[P3M])*x[NM_] +
           (x[Q0M]+T(2)*x[Q1M]+T(3)*x[Q2M])*x[NM_]
    ExaModels.@add_obj(c, (hinn - NINN)^2 / T(1))

    # Equality constraints (E type, identity group function, no scale)
    # AGE2: NC2 + NA2 = N2
    ExaModels.@add_con(c, x[NC2] + x[NA2] - N2)
    # AGE3: NC3 + NA3 = N3
    ExaModels.@add_con(c, x[NC3] + x[NA3] - N3)
    # PSF: P1F+P2F+P3F+P4F+P5F = 1
    ExaModels.@add_con(c, x[P1F]+x[P2F]+x[P3F]+x[P4F]+x[P5F] - T(1))
    # PSW: P1W+P2W+P3W = 1
    ExaModels.@add_con(c, x[P1W]+x[P2W]+x[P3W] - T(1))
    # PSM: P1M+P2M+P3M = 1
    ExaModels.@add_con(c, x[P1M]+x[P2M]+x[P3M] - T(1))
    # QSF: Q0F+Q1F+Q2F = 1
    ExaModels.@add_con(c, x[Q0F]+x[Q1F]+x[Q2F] - T(1))
    # QSW: Q0W+Q1W+Q2W = 1
    ExaModels.@add_con(c, x[Q0W]+x[Q1W]+x[Q2W] - T(1))
    # QSM: Q0M+Q1M+Q2M = 1
    ExaModels.@add_con(c, x[Q0M]+x[Q1M]+x[Q2M] - T(1))

    # Inequality constraints (L type ≤ 0)
    # INEQ2: NC2 - NA2 ≤ 0
    ExaModels.@add_con(c, x[NC2] - x[NA2]; lcon = T(-Inf), ucon = T(0))
    # INEQ3: NC3 - NA3 ≤ 0
    ExaModels.@add_con(c, x[NC3] - x[NA3]; lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
