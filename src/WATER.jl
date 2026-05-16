function water_model(; T = Float64, backend = nothing, kwargs...)
    # WATER: small nonlinear water distribution network
    # 31 variables, all ≥ 0, ≤ 1200
    # 8 obj groups (POWER type: v^2.852 / sigma^1.852), 1 linear obj group, 10 equality constraints
    # Variable numbering (1-indexed):
    # 1=Q0102, 2=Q0203, 3=Q0204, 4=Q0305, 5=Q0405, 6=Q0406, 7=Q0607, 8=Q0705,
    # 9=Q01_0, 10=y02_p, 11=y03_p, 12=y04_p, 13=y05_p, 14=y06_p, 15=y07_p,
    # 16=yq_02, 17=yq_03, 18=yq_04, 19=yq_05, 20=yq_06, 21=yq_07,
    # 22=Q0201, 23=Q0302, 24=Q0402, 25=Q0503, 26=Q0504, 27=Q0604, 28=Q0507, 29=Q0706,
    # 30=y_p_0, 31=y_0_q

    # SCALE values for the 8 POWER groups
    sigma = T[105665.6, 3613.412, 105665.6, 890.1553, 76.66088, 55145.82, 26030.46, 890.1553]

    # The 8 POWER groups each have a linear coefficient of 1.0 from the corresponding variable
    # (Q0102 → obj0102, Q0203 → obj0203, etc.)
    # GROUP TYPE POWER: phi(alpha) = alpha^2.852
    # contribution = sigma * phi(v / sigma) = sigma * (v/sigma)^2.852 = v^2.852 / sigma^1.852

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 31; lvar = T(0), uvar = T(1200), start = T(0))

    # Objective: 8 POWER terms + 1 linear term (obj group)
    # POWER term for each of the 8 flow variables (indices 1..8): v = x[i], σ = sigma[i]
    # contribution = sigma * (v/sigma)^2.852 = v^2.852 / sigma^1.852
    sig18 = T[sigma[i]^1.852 for i in 1:8]  # precompute sigma^1.852
    ExaModels.@add_par(c, sig_par, sig18)
    ExaModels.@add_obj(c, x[i]^T(2.852) / sig_par[i] for i in 1:8)

    # Linear obj group (group "obj", no type, identity):
    # obj += 210*Q01_0 + 210*y02_p + 210*y03_p + 210*y04_p + 210*y05_p + 210*y06_p + 210*y07_p
    #      - 175*yq_02 - 190*yq_03 - 185*yq_04 - 180*yq_05 - 195*yq_06 - 190*yq_07
    ExaModels.@add_obj(c,
        T(210)*(x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15])
        - T(175)*x[16] - T(190)*x[17] - T(185)*x[18] - T(180)*x[19] - T(195)*x[20] - T(190)*x[21])

    # Equality constraints (10 E-type groups with RHS):
    # c1: Q0102 + Q01_0 - Q0201 = 1120
    ExaModels.@add_con(c, x[1] + x[9] - x[22] - T(1120))

    # c2: -Q0102 + Q0203 + Q0204 + y02_p - yq_02 - Q0302 - Q0402 = -100
    ExaModels.@add_con(c, -x[1] + x[2] + x[3] + x[10] - x[16] - x[23] - x[24] - T(-100))

    # c3: -Q0203 + Q0305 + y03_p - yq_03 - Q0503 = -100
    ExaModels.@add_con(c, -x[2] + x[4] + x[11] - x[17] - x[25] - T(-100))

    # c4: -Q0204 + Q0405 + Q0406 + y04_p - yq_04 - Q0504 = -120
    ExaModels.@add_con(c, -x[3] + x[5] + x[6] + x[12] - x[18] - x[26] - T(-120))

    # c5: -Q0305 - Q0405 + Q0705 + y05_p - yq_05 - Q0503 + Q0507 - Q0504 = -270
    # Wait, from COLUMNS:
    # Q0305 → c3:+1, c5:-1... No: Q0305 appears in obj0305 and c3:1, c5:-1
    # Let me re-read: c5 gets: -Q0305(-1)? No, Q0305 c3:+1 c5:-1
    # Actually: Q0305 → c3:+1 (outflow from node 3), c5:-1 (inflow to node 5)
    # c5: -Q0305 - Q0405 - Q0705 + y05_p - yq_05 + Q0503 + Q0504 - Q0507 = -270
    # Careful re-reading the COLUMNS section:
    # Q0305: obj0305=1, c3=1, c5=-1
    # Q0405: obj0405=1, c4=1, c5=-1
    # Q0705: obj0705=1, c5=-1, c7=1
    # y05_p: obj=210, c5=1, c9=-1
    # yq_05: obj=-180, c5=-1, c10=1
    # Q0503: c3=-1, c5=1
    # Q0504: c4=-1, c5=1
    # Q0507: c5=1, c7=-1
    # c5 = -Q0305 - Q0405 - Q0705 + y05_p - yq_05 + Q0503 + Q0504 + Q0507 = -270
    ExaModels.@add_con(c, -x[4] - x[5] - x[8] + x[13] - x[19] + x[25] + x[26] + x[28] - T(-270))

    # c6: -Q0406 + Q0606? No, Q0607...
    # Q0406: obj0406=1, c4=1, c6=-1
    # Q0607: obj0607=1, c6=1, c7=-1
    # y06_p: obj=210, c6=1, c9=-1
    # yq_06: obj=-195, c6=-1, c10=1
    # Q0604: c4=-1, c6=1  (Wait: Q0604 is column 27)
    # Actually COLUMNS: Q0604 → c4=-1, c6=1
    # c6: -Q0406 + Q0607 + y06_p - yq_06 + Q0604 = -330
    ExaModels.@add_con(c, -x[6] + x[7] + x[14] - x[20] + x[27] - T(-330))

    # c7: -Q0607 + Q0705 + y07_p - yq_07 + Q0706 - Q0507 = -200
    # Q0607: c6=1, c7=-1 → c7: -Q0607
    # Q0705: c5=-1, c7=1 → c7: +Q0705
    # y07_p: obj=210, c7=1, c9=-1 → c7: +y07_p
    # yq_07: obj=-190, c7=-1, c10=1 → c7: -yq_07
    # Q0706: c6=-1, c7=1 → c7: +Q0706
    # Q0507: c5=1, c7=-1 → c7: -Q0507
    ExaModels.@add_con(c, -x[7] + x[8] + x[15] - x[21] + x[29] - x[28] - T(-200))

    # c8: Q01_0 - y_p_0 + y_0_q = 0
    # Q01_0: c1=1, c8=-1 → c8: -Q01_0
    # y_p_0: c8=-1, c9=1 → c8: -y_p_0? Wait: y_p_0: c8=-1, c9=1
    # Actually COLUMNS:
    # Q01_0 → obj=210, c1=1, c8=-1
    # y_p_0 → c8=-1, c9=1
    # y_0_q → c8=1, c10=-1
    # c8: -Q01_0 - y_p_0 + y_0_q = 0
    ExaModels.@add_con(c, -x[9] - x[30] + x[31])

    # c9: -y02_p - y03_p - y04_p - y05_p - y06_p - y07_p + y_p_0 = 0
    # y02_p → c2=1, c9=-1
    # y03_p → c3=1, c9=-1
    # ... all y[k]_p → c9=-1
    # y_p_0 → c8=-1, c9=1
    ExaModels.@add_con(c, -x[10] - x[11] - x[12] - x[13] - x[14] - x[15] + x[30])

    # c10: yq_02 + yq_03 + yq_04 + yq_05 + yq_06 + yq_07 - y_0_q = 0
    # yq_02 → c10=1; ... yq_07 → c10=1; y_0_q → c8=1, c10=-1
    ExaModels.@add_con(c, x[16] + x[17] + x[18] + x[19] + x[20] + x[21] - x[31])

    return ExaModels.ExaModel(c; kwargs...)
end
