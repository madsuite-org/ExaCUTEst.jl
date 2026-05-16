# DNIEPER: River Dnieper water planning problem
# 61 variables (56 X(I) + 4 fixed boundary vars + AC), 24 equality constraints
# Objective: maximize water use (converted to min via SCALE=-1 → same sign convention)
# Constraints: polynomial equality constraints using WJ/WK cubic elements
function dnieper_model(; T = Float64, backend = nothing, kwargs...)
    # Problem parameters C(1..24)
    C = T[5.61, 4.68, 1.62, 1.8, 2.13, 2.1, 1.99, 2.02,
          2.14, 2.15, 2.36, 2.63, -0.02, -0.01, -0.16, -0.47,
          -0.75, -0.94, -0.93, -0.99, -0.42, -0.07, 0.04, -0.06]

    # WJ element constants (cubic polynomial, used for I=1..12 constraints)
    WJA1 = T(34.547); WJA2 = T(-0.55878); WJA3 = T(8.05339)
    WJA4 = T(-0.02252); WJA5 = T(-0.29316); WJA6 = T(-0.013521)
    WJA7 = T(0.00042); WJA8 = T(0.00267); WJA9 = T(0.000281)
    WJA10 = T(0.0000032)

    # WK element constants (cubic polynomial, no A6, used for I=13..24 constraints)
    WKA1 = T(20.923); WKA2 = T(-4.22088); WKA3 = T(1.42061)
    WKA4 = T(-0.41040); WKA5 = T(-0.15082)
    WKA7 = T(-0.00826); WKA8 = T(0.00404); WKA9 = T(0.000168)
    WKA10 = T(-0.000038)

    # Variable indexing (61 variables total):
    # X(1..56) -> Julia index 1..56
    # X0F      -> Julia index 57 (fixed = 50.82)
    # X24F     -> Julia index 58 (fixed = 2.0)
    # X12F     -> Julia index 59 (fixed = 15.5)
    # X36F     -> Julia index 60 (fixed = 2.3)
    # AC       -> Julia index 61 (free)
    NV = 61
    IDX_X0F  = 57; IDX_X24F = 58; IDX_X12F = 59; IDX_X36F = 60; IDX_AC = 61

    lvar = fill(T(0), NV)
    uvar = fill(T(Inf), NV)
    start_vals = fill(T(0), NV)

    # X(1..12): 51.2 <= X(I) <= 51.4, start=51.35
    for i in 1:12
        lvar[i] = T(51.2); uvar[i] = T(51.4); start_vals[i] = T(51.35)
    end
    # X(13..24): 15.0 <= X(I) <= 16.1, start=15.5
    for i in 13:24
        lvar[i] = T(15.0); uvar[i] = T(16.1); start_vals[i] = T(15.5)
    end
    # X(25..36): 0.4 <= X(I) <= 4.6, start=2.5
    for i in 25:36
        lvar[i] = T(0.4); uvar[i] = T(4.6); start_vals[i] = T(2.5)
    end
    # X(37..48): 0.5 <= X(I) <= 4.8, start=2.6
    for i in 37:48
        lvar[i] = T(0.5); uvar[i] = T(4.8); start_vals[i] = T(2.6)
    end
    # X(49..56): 0.0 <= X(I) <= 0.7, start=0.3
    for i in 49:56
        lvar[i] = T(0.0); uvar[i] = T(0.7); start_vals[i] = T(0.3)
    end
    # X0F: fixed at 50.82
    lvar[IDX_X0F] = T(50.82); uvar[IDX_X0F] = T(50.82); start_vals[IDX_X0F] = T(50.82)
    # X24F: fixed at 2.0
    lvar[IDX_X24F] = T(2.0); uvar[IDX_X24F] = T(2.0); start_vals[IDX_X24F] = T(2.0)
    # X12F: fixed at 15.5
    lvar[IDX_X12F] = T(15.5); uvar[IDX_X12F] = T(15.5); start_vals[IDX_X12F] = T(15.5)
    # X36F: fixed at 2.3
    lvar[IDX_X36F] = T(2.3); uvar[IDX_X36F] = T(2.3); start_vals[IDX_X36F] = T(2.3)
    # AC: free
    lvar[IDX_AC] = T(-Inf); uvar[IDX_AC] = T(Inf); start_vals[IDX_AC] = T(0.0)

    ec = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(ec, x, NV; lvar = lvar, uvar = uvar, start = start_vals)

    # Objective: OBJ group (identity type, SCALE=-1 → contribution negated for minimization)
    # v = linear_part + element_part - CONSTANT(-112.464) = ... + 112.464
    # With SCALE=-1: contribution = -1 * F(v/(-1)) = -1 * (v/(-1)) = v? No:
    # σ*F(v/σ) = -1 * F(-v) = -1 * (-v) = v (for identity F)
    # BUT CUTEst convention: for minimization, if problem is 'maximize f', CUTEst returns -f.
    # The SCALE=-1 in XN OBJ 'SCALE' -1.0 negates the group's contribution to the objective.
    # In SIF for a maximization problem: contribution = SCALE * v = -1 * v (i.e., negate to minimize)
    # So the objective value in CUTEst = -(water_use_function)
    # Linear terms for I=1..12:
    #   +19.95*x[I+24] + 0.07656*x[I] - 24.89*x[I+36] - 0.7135*x[I+12]
    # These come from XN OBJ ... and are part of group value v.
    # With SCALE=-1: objective contribution = -v
    # So objective = -(linear_terms + elements + 112.464)
    # = -sum_I (19.95*x[I+24] + 0.07656*x[I] - 24.89*x[I+36] - 0.7135*x[I+12])
    #   - sum_I 2.155*x[I+12]*x[I+36] + 2000*x[AC]^2 - 112.464

    # OBJ group: identity type, no GROUP TYPE
    # Per CLAUDE.md: SCALE is ignored for identity objective groups → contribution = v
    # v = linear_part + element_part - CONSTANT(-112.464) = ... + 112.464
    # Linear terms for I=1..12: 19.95*x[I+24] + 0.07656*x[I] - 24.89*x[I+36] - 0.7135*x[I+12]
    # Elements: E(I) = 2PR(x[I+12], x[I+36]) coef 2.155; ACSQ = SQ(AC) coef -2000.0
    obj_data = [(xi = i, xi12 = i+12, xi24 = i+24, xi36 = i+36) for i in 1:12]
    ExaModels.@add_obj(ec,
        T(19.95)*x[d.xi24] + T(0.07656)*x[d.xi] +
        T(-24.89)*x[d.xi36] + T(-0.7135)*x[d.xi12] +
        T(2.155)*x[d.xi12]*x[d.xi36]
        for d in obj_data)
    # ACSQ element: SQ(AC) = AC² with coefficient -2000.0
    ExaModels.@add_obj(ec, T(-2000.0) * x[IDX_AC]^2 for _ in 1:1)
    ExaModels.@add_obj(ec, T(112.464) for _ in 1:1)

    # Constraints CC(I) = 0 for I=1..24
    # v = linear_part + W1(I) - W2(I) - CONSTANT(-C(I)) = linear + W1(I) - W2(I) + C(I) = 0

    # W2(1): WJ(X0F, X24F) — both fixed, scalar constant
    W2_1_const = WJA1 + WJA2*T(50.82) + WJA3*T(2.0) + WJA4*T(50.82)^2 +
                 WJA5*T(50.82)*T(2.0) + WJA6*T(2.0)^2 + WJA7*T(50.82)^3 +
                 WJA8*T(50.82)^2*T(2.0) + WJA9*T(50.82)*T(2.0)^2 + WJA10*T(2.0)^3

    # CC(1): -2.68*x[25] + WJ(x[1],x[25]) - W2_1_const + C[1] = 0
    ExaModels.@add_con(ec,
        T(-2.68)*x[25] +
        WJA1 + WJA2*x[1] + WJA3*x[25] + WJA4*x[1]^2 + WJA5*x[1]*x[25] +
        WJA6*x[25]^2 + WJA7*x[1]^3 + WJA8*x[1]^2*x[25] + WJA9*x[1]*x[25]^2 +
        WJA10*x[25]^3 - W2_1_const + C[1])

    # CC(I) for I=2..4: -2.68*x[I+24] + WJ(x[I],x[I+24]) - WJ(x[I-1],x[I+23]) + C[I] = 0
    con24_data = [(xi = i, xip24 = i+24, xim1 = i-1, xim1p24 = i+23, ci = C[i])
                  for i in 2:4]
    ExaModels.@add_con(ec,
        T(-2.68)*x[d.xip24] +
        WJA1 + WJA2*x[d.xi] + WJA3*x[d.xip24] + WJA4*x[d.xi]^2 +
        WJA5*x[d.xi]*x[d.xip24] + WJA6*x[d.xip24]^2 + WJA7*x[d.xi]^3 +
        WJA8*x[d.xi]^2*x[d.xip24] + WJA9*x[d.xi]*x[d.xip24]^2 + WJA10*x[d.xip24]^3 -
        (WJA1 + WJA2*x[d.xim1] + WJA3*x[d.xim1p24] + WJA4*x[d.xim1]^2 +
         WJA5*x[d.xim1]*x[d.xim1p24] + WJA6*x[d.xim1p24]^2 + WJA7*x[d.xim1]^3 +
         WJA8*x[d.xim1]^2*x[d.xim1p24] + WJA9*x[d.xim1]*x[d.xim1p24]^2 +
         WJA10*x[d.xim1p24]^3) + d.ci
        for d in con24_data)

    # CC(I) for I=5..8: -2.68*x[I+24] - 2.68*x[I+44] + WJ(x[I],x[I+24]) - WJ(x[I-1],x[I+23]) + C[I] = 0
    # x[I+44] for I=5..8: x[49..52]
    con58_data = [(xi = i, xip24 = i+24, xip44 = i+44, xim1 = i-1, xim1p24 = i+23, ci = C[i])
                  for i in 5:8]
    ExaModels.@add_con(ec,
        T(-2.68)*x[d.xip24] + T(-2.68)*x[d.xip44] +
        WJA1 + WJA2*x[d.xi] + WJA3*x[d.xip24] + WJA4*x[d.xi]^2 +
        WJA5*x[d.xi]*x[d.xip24] + WJA6*x[d.xip24]^2 + WJA7*x[d.xi]^3 +
        WJA8*x[d.xi]^2*x[d.xip24] + WJA9*x[d.xi]*x[d.xip24]^2 + WJA10*x[d.xip24]^3 -
        (WJA1 + WJA2*x[d.xim1] + WJA3*x[d.xim1p24] + WJA4*x[d.xim1]^2 +
         WJA5*x[d.xim1]*x[d.xim1p24] + WJA6*x[d.xim1p24]^2 + WJA7*x[d.xim1]^3 +
         WJA8*x[d.xim1]^2*x[d.xim1p24] + WJA9*x[d.xim1]*x[d.xim1p24]^2 +
         WJA10*x[d.xim1p24]^3) + d.ci
        for d in con58_data)

    # CC(I) for I=9..12: -2.68*x[I+24] + WJ(x[I],x[I+24]) - WJ(x[I-1],x[I+23]) + C[I] = 0
    con912_data = [(xi = i, xip24 = i+24, xim1 = i-1, xim1p24 = i+23, ci = C[i])
                   for i in 9:12]
    ExaModels.@add_con(ec,
        T(-2.68)*x[d.xip24] +
        WJA1 + WJA2*x[d.xi] + WJA3*x[d.xip24] + WJA4*x[d.xi]^2 +
        WJA5*x[d.xi]*x[d.xip24] + WJA6*x[d.xip24]^2 + WJA7*x[d.xi]^3 +
        WJA8*x[d.xi]^2*x[d.xip24] + WJA9*x[d.xi]*x[d.xip24]^2 + WJA10*x[d.xip24]^3 -
        (WJA1 + WJA2*x[d.xim1] + WJA3*x[d.xim1p24] + WJA4*x[d.xim1]^2 +
         WJA5*x[d.xim1]*x[d.xim1p24] + WJA6*x[d.xim1p24]^2 + WJA7*x[d.xim1]^3 +
         WJA8*x[d.xim1]^2*x[d.xim1p24] + WJA9*x[d.xim1]*x[d.xim1p24]^2 +
         WJA10*x[d.xim1p24]^3) + d.ci
        for d in con912_data)

    # W2(13): WK(X12F=15.5, X36F=2.3) — both fixed
    W2_13_const = WKA1 + WKA2*T(15.5) + WKA3*T(2.3) + WKA4*T(15.5)^2 +
                  WKA5*T(15.5)*T(2.3) + WKA7*T(15.5)^3 +
                  WKA8*T(15.5)^2*T(2.3) + WKA9*T(15.5)*T(2.3)^2 + WKA10*T(2.3)^3

    # CC(13): -2.68*x[25] - 2.68*x[37] - x[AC] + WK(x[13],x[37]) - W2_13_const + C[13] = 0
    # Note: x[I+12] for I=13: x[25]; x[I+24] for I=13: x[37]
    ExaModels.@add_con(ec,
        T(-2.68)*x[25] + T(-2.68)*x[37] - x[IDX_AC] +
        WKA1 + WKA2*x[13] + WKA3*x[37] + WKA4*x[13]^2 + WKA5*x[13]*x[37] +
        WKA7*x[13]^3 + WKA8*x[13]^2*x[37] + WKA9*x[13]*x[37]^2 + WKA10*x[37]^3 -
        W2_13_const + C[13])

    # CC(I) for I=14..16: -2.68*x[I+12] - 2.68*x[I+24] - x[AC] + WK(x[I],x[I+24]) - WK(x[I-1],x[I+23]) + C[I] = 0
    # Note: x[I+24] for I=13..24 corresponds to X(I+24) = y variable (reservoir discharge)
    #       x[I+12] for I=13: x[25], I=14: x[26], etc.
    con1416_data = [(xi = i, xip12 = i+12, xip24 = i+24, xim1 = i-1, xim1p24 = i+23, ci = C[i])
                    for i in 14:16]
    ExaModels.@add_con(ec,
        T(-2.68)*x[d.xip12] + T(-2.68)*x[d.xip24] - x[IDX_AC] +
        WKA1 + WKA2*x[d.xi] + WKA3*x[d.xip24] + WKA4*x[d.xi]^2 +
        WKA5*x[d.xi]*x[d.xip24] + WKA7*x[d.xi]^3 +
        WKA8*x[d.xi]^2*x[d.xip24] + WKA9*x[d.xi]*x[d.xip24]^2 + WKA10*x[d.xip24]^3 -
        (WKA1 + WKA2*x[d.xim1] + WKA3*x[d.xim1p24] + WKA4*x[d.xim1]^2 +
         WKA5*x[d.xim1]*x[d.xim1p24] + WKA7*x[d.xim1]^3 +
         WKA8*x[d.xim1]^2*x[d.xim1p24] + WKA9*x[d.xim1]*x[d.xim1p24]^2 +
         WKA10*x[d.xim1p24]^3) + d.ci
        for d in con1416_data)

    # CC(I) for I=17..20: -2.68*x[I+12] - 2.68*x[I+24] - 2.68*x[I+36] - x[AC] + WK(x[I],x[I+24]) - WK(x[I-1],x[I+23]) + C[I] = 0
    # x[I+36] for I=17..20: x[53..56]
    con1720_data = [(xi = i, xip12 = i+12, xip24 = i+24, xip36 = i+36,
                     xim1 = i-1, xim1p24 = i+23, ci = C[i]) for i in 17:20]
    ExaModels.@add_con(ec,
        T(-2.68)*x[d.xip12] + T(-2.68)*x[d.xip24] + T(-2.68)*x[d.xip36] - x[IDX_AC] +
        WKA1 + WKA2*x[d.xi] + WKA3*x[d.xip24] + WKA4*x[d.xi]^2 +
        WKA5*x[d.xi]*x[d.xip24] + WKA7*x[d.xi]^3 +
        WKA8*x[d.xi]^2*x[d.xip24] + WKA9*x[d.xi]*x[d.xip24]^2 + WKA10*x[d.xip24]^3 -
        (WKA1 + WKA2*x[d.xim1] + WKA3*x[d.xim1p24] + WKA4*x[d.xim1]^2 +
         WKA5*x[d.xim1]*x[d.xim1p24] + WKA7*x[d.xim1]^3 +
         WKA8*x[d.xim1]^2*x[d.xim1p24] + WKA9*x[d.xim1]*x[d.xim1p24]^2 +
         WKA10*x[d.xim1p24]^3) + d.ci
        for d in con1720_data)

    # CC(I) for I=21..24: -2.68*x[I+12] - 2.68*x[I+24] - x[AC] + WK(x[I],x[I+24]) - WK(x[I-1],x[I+23]) + C[I] = 0
    # x[I+12] for I=21..24: x[33..36]; x[I+24] for I=21..24: x[45..48]
    con2124_data = [(xi = i, xip12 = i+12, xip24 = i+24,
                     xim1 = i-1, xim1p24 = i+23, ci = C[i]) for i in 21:24]
    ExaModels.@add_con(ec,
        T(-2.68)*x[d.xip12] + T(-2.68)*x[d.xip24] - x[IDX_AC] +
        WKA1 + WKA2*x[d.xi] + WKA3*x[d.xip24] + WKA4*x[d.xi]^2 +
        WKA5*x[d.xi]*x[d.xip24] + WKA7*x[d.xi]^3 +
        WKA8*x[d.xi]^2*x[d.xip24] + WKA9*x[d.xi]*x[d.xip24]^2 + WKA10*x[d.xip24]^3 -
        (WKA1 + WKA2*x[d.xim1] + WKA3*x[d.xim1p24] + WKA4*x[d.xim1]^2 +
         WKA5*x[d.xim1]*x[d.xim1p24] + WKA7*x[d.xim1]^3 +
         WKA8*x[d.xim1]^2*x[d.xim1p24] + WKA9*x[d.xim1]*x[d.xim1p24]^2 +
         WKA10*x[d.xim1p24]^3) + d.ci
        for d in con2124_data)

    return ExaModels.ExaModel(ec; kwargs...)
end
