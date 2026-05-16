# ZAMB2-9: Optimal control of Zambezi hydropower system
# NY=1 year, YEAR=1960, N=12 months
# 138 variables (6 fixed: V1(0),V2(0),V3(0),V4(0),F2-1,F2-2), 48 equality constraints
# Differs from ZAMB2-8: PHI=100, V20=V2REF=4.5, inflow from YEAR=1960
# SIF: OOR2-RN-138-48. Source: Arnold et al. (1994), JOTA 81(2), 221-248.
function zamb2_9_model(; T = Float64, backend = nothing, kwargs...)
    N = 12

    SC    = T(1e-5)
    ALFA  = T(1.2285) * SC
    PHI   = T(100.0)  * SC
    PSI   = T(100.0)  * SC
    PHI3  = T(20.0)

    V1REF = T(50.0);  V2REF = T(4.5);  V3REF = T(2.5);  V4REF = T(45.0)
    Q1REF = T(3.8);   F2REF = T(0.6);  Q3REF = T(0.544); Q4REF = T(4.78)
    V10   = T(50.0);  V20   = T(4.5);  V30   = T(2.5);   V40   = T(45.0)
    IN3   = T(-0.0388)

    # kk_base = (1960-1930)*12 = 360; kk = 360+k for k=0..11
    IN1 = T[0.818, 0.918, 1.835, 2.970, 3.362, 5.450,
            14.241, 11.901, 6.592, 4.130, 2.351, 1.468]
    IN2 = T[0.127, 0.084, 0.056, 0.057, 0.093, 0.396,
            1.306, 3.348, 2.948, 1.587, 0.664, 0.368]
    IN4 = T[-0.155, 0.191, 3.203, 4.238, 6.402, 2.096,
             0.731, 0.342, 0.367, 0.278, 0.285, 0.176]

    E1 = T[0.1672, 0.1158, 0.0, 0.0, 0.0, 0.0074,
           0.0910, 0.1113, 0.0994, 0.0865, 0.1113, 0.1464]
    E4 = T[0.1710, 0.1242, 0.0, 0.0, 0.0, 0.0404,
           0.0989, 0.1119, 0.1011, 0.0866, 0.1128, 0.1471]

    FA1_A = T(-0.0000553);  FA1_B = T(0.02232);  FA1_C = T(4.361)
    FA2_A = T(-0.0026236);  FA2_B = T(0.07353);  FA2_C = T(0.036)
    FA3_A = T(-0.2653036);  FA3_B = T(1.42945);  FA3_C = T(0.214)
    FA4_A = T(-0.0001706);  FA4_B = T(0.04366);  FA4_C = T(0.861)

    FHQ1_A = T(-0.000372);   FHQ1_B = T(0.22463);  FHQ1_C = T(97.559)
    FHQ3_A = T(-0.399511);   FHQ3_B = T(2.16826);  FHQ3_C = T(393.914)
    FHQ4_A = T(-0.005591);   FHQ4_B = T(0.88079);  FHQ4_C = T(90.576)

    iV1(k) = 4k+1;  iV2(k) = 4k+2;  iV3(k) = 4k+3;  iV4(k) = 4k+4
    iQ1(k) = 52+7k+1; iQ3(k) = 52+7k+2; iQ4(k) = 52+7k+3
    iF1(k) = 52+7k+4; iF2(k) = 52+7k+5; iF3(k) = 52+7k+6; iF4(k) = 52+7k+7
    iF2m1  = 137;  iF2m2 = 138

    nvar = 138
    lvar = fill(T(-Inf), nvar);  uvar = fill(T(Inf), nvar);  x0 = fill(T(0), nvar)

    lvar[iV1(0)] = V10;  uvar[iV1(0)] = V10;  x0[iV1(0)] = V10
    lvar[iV2(0)] = V20;  uvar[iV2(0)] = V20;  x0[iV2(0)] = V20
    lvar[iV3(0)] = V30;  uvar[iV3(0)] = V30;  x0[iV3(0)] = V30
    lvar[iV4(0)] = V40;  uvar[iV4(0)] = V40;  x0[iV4(0)] = V40

    for k in 1:N
        lvar[iV1(k)] = T(0.05);  uvar[iV1(k)] = T(70.97);  x0[iV1(k)] = V10
        lvar[iV2(k)] = T(0.7);   uvar[iV2(k)] = T(5.75);   x0[iV2(k)] = V20
        lvar[iV3(k)] = T(0.02);  uvar[iV3(k)] = T(3.0);    x0[iV3(k)] = V30
        lvar[iV4(k)] = T(0.0);   uvar[iV4(k)] = T(61.8);   x0[iV4(k)] = V40
    end

    for k in 0:N-1
        lvar[iQ1(k)] = T(2.1);   uvar[iQ1(k)] = T(3.8);    x0[iQ1(k)] = IN1[k+1]
        lvar[iQ3(k)] = T(0.3);   uvar[iQ3(k)] = T(0.544);  x0[iQ3(k)] = IN2[k+1]
        lvar[iQ4(k)] = T(3.59);  uvar[iQ4(k)] = T(4.78);   x0[iQ4(k)] = IN1[k+1]
        lvar[iF1(k)] = T(0.0);   uvar[iF1(k)] = T(8.1);    x0[iF1(k)] = T(0)
        uvar[iF2(k)] = T(10.9);                              x0[iF2(k)] = IN2[k+1]
        lvar[iF3(k)] = T(0.0);   uvar[iF3(k)] = T(10.9);   x0[iF3(k)] = T(0)
        lvar[iF4(k)] = T(0.0);   uvar[iF4(k)] = T(36.1);   x0[iF4(k)] = T(0)
    end

    for k in 0:3;  lvar[iF2(k)] = T(0.37); end
    lvar[iF2(4)] = T(0.555);  lvar[iF2(5)] = T(0.8325);  lvar[iF2(6)] = T(0.555)
    for k in 7:11; lvar[iF2(k)] = T(0.37); end

    lvar[iF2m1] = F2REF;  uvar[iF2m1] = F2REF;  x0[iF2m1] = F2REF
    lvar[iF2m2] = F2REF;  uvar[iF2m2] = F2REF;  x0[iF2m2] = F2REF

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = x0)

    ExaModels.@add_par(c, k_data, [(
        v1  = iV1(k), v1n = iV1(k+1), q1 = iQ1(k),
        v3  = iV3(k), v3n = iV3(k+1), q3 = iQ3(k),
        v4  = iV4(k), v4n = iV4(k+1), q4 = iQ4(k),
        f2  = iF2(k),
    ) for k in 0:N-1])

    ExaModels.@add_obj(c,
        (-ALFA) * (
            x[d.q1] * (FHQ1_A*(x[d.v1]^2 + x[d.v1n]^2) + FHQ1_B*(x[d.v1] + x[d.v1n]) + T(2)*FHQ1_C) +
            x[d.q3] * (FHQ3_A*(x[d.v3]^2 + x[d.v3n]^2) + FHQ3_B*(x[d.v3] + x[d.v3n]) + T(2)*FHQ3_C) +
            x[d.q4] * (FHQ4_A*(x[d.v4]^2 + x[d.v4n]^2) + FHQ4_B*(x[d.v4] + x[d.v4n]) + T(2)*FHQ4_C)
        ) + PSI * (
            (x[d.q1] - Q1REF)^2 + (x[d.f2] - F2REF)^2 +
            (x[d.q3] - Q3REF)^2 + (x[d.q4] - Q4REF)^2
        )
        for d in k_data)

    ExaModels.@add_obj(c,
        PHI * (x[iV1(N)] - V1REF)^2 + PHI * (x[iV2(N)] - V2REF)^2 +
        (PHI * PHI3) * (x[iV3(N)] - V3REF)^2 + PHI * (x[iV4(N)] - V4REF)^2)

    f2del = vcat([iF2m2, iF2m1], [iF2(k) for k in 0:N-3])

    for k in 0:N-1
        e1  = E1[(k % 12) + 1];  e4 = E4[(k % 12) + 1]
        in1 = IN1[k+1]; in2 = IN2[k+1]; in4 = IN4[k+1]
        v1  = iV1(k); v1n = iV1(k+1); q1 = iQ1(k); f1 = iF1(k)
        v2  = iV2(k); v2n = iV2(k+1); f2 = iF2(k)
        v3  = iV3(k); v3n = iV3(k+1); q3 = iQ3(k); f3 = iF3(k)
        f2d = f2del[k+1]
        v4  = iV4(k); v4n = iV4(k+1); q4 = iQ4(k); f4 = iF4(k)

        ExaModels.@add_con(c,
            -x[v1] + x[q1] + x[f1] + x[v1n] +
            (FA1_A*x[v1]^2 + FA1_B*x[v1] + FA1_C)*e1 - in1)
        ExaModels.@add_con(c,
            -x[v2] + x[f2] + x[v2n] +
            (FA2_A*x[v2]^2 + FA2_B*x[v2] + FA2_C)*e1 - in2)
        ExaModels.@add_con(c,
            -x[v3] - x[f2d] + x[q3] + x[f3] + x[v3n] +
            (FA3_A*x[v3]^2 + FA3_B*x[v3] + FA3_C)*e1 - IN3)
        ExaModels.@add_con(c,
            -x[v4] - x[q1] - x[f1] - x[q3] - x[f3] +
             x[q4] + x[f4] + x[v4n] +
            (FA4_A*x[v4]^2 + FA4_B*x[v4] + FA4_C)*e4 - in4)
    end

    return ExaModels.ExaModel(c; kwargs...)
end
