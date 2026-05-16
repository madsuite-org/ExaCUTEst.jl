# SYNTHES3: Synthesis of processing system 3 (MINLP treated as NLP)
# Source: Duran & Grossmann, Math. Prog. 36 (1986), pp. 307-339, problem 3
# 17 vars: X1..X9 continuous, Y1..Y8 INTEGER→continuous
# x[1..9]=X(1..9), x[10..17]=Y(1..8)
# CUTEst constraint ordering determined empirically (matches sifdecoder):
# L16(E), L18(E), L17(L), L19(L), L1..L15(L), N4, N3, N2, N1
function synthes3_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 17;
        lvar = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        uvar = T[2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 3.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # x[1..9]=X(1..9), x[10..17]=Y(1..8)
    # Objective: constant rhs=-120 → +120; elements: exp(X1)+exp(X2/1.2)-65*log(X3+X4+1)-90*log(X5+1)-80*log(X6+1)
    ExaModels.@add_obj(c, 5 * x[10] + 8 * x[11] + 6 * x[12] + 10 * x[13] + 6 * x[14] +
                          7 * x[15] + 4 * x[16] + 5 * x[17] -
                          10 * x[1] - 15 * x[2] + 15 * x[3] + 80 * x[4] + 25 * x[5] +
                          35 * x[6] - 40 * x[7] + 15 * x[8] - 35 * x[9] + 120 +
                          exp(x[1]) + exp(x[2] / T(1.2)) -
                          65 * log(x[3] + x[4] + 1) - 90 * log(x[5] + 1) - 80 * log(x[6] + 1))
    # Constraint ordering (empirically determined from CUTEst sifdecoder):
    # cons[1] = L16 (E): Y1+Y2-1 = 0
    ExaModels.@add_con(c, x[10] + x[11] - 1)
    # cons[2] = L18 (E): -Y4+Y6+Y7 = 0
    ExaModels.@add_con(c, -x[13] + x[15] + x[16])
    # cons[3] = L17 (L<=0): Y4+Y5-1 <= 0
    ExaModels.@add_con(c, x[13] + x[14] - 1; lcon = T(-Inf), ucon = T(0))
    # cons[4] = L19 (L<=0): Y3-Y8 <= 0
    ExaModels.@add_con(c, x[12] - x[17]; lcon = T(-Inf), ucon = T(0))
    # cons[5] = L1 (L<=0): -X1-X2+X3+2X4+0.8X5+0.8X6-0.5X7-X8-2X9 <= 0
    ExaModels.@add_con(c, -x[1] - x[2] + x[3] + 2 * x[4] + T(0.8) * x[5] + T(0.8) * x[6] - T(0.5) * x[7] - x[8] - 2 * x[9]; lcon = T(-Inf), ucon = T(0))
    # cons[6] = L2 (L<=0): -X1-X2+2X4+0.8X5+0.8X6-2X7-X8-2X9 <= 0
    ExaModels.@add_con(c, -x[1] - x[2] + 2 * x[4] + T(0.8) * x[5] + T(0.8) * x[6] - 2 * x[7] - x[8] - 2 * x[9]; lcon = T(-Inf), ucon = T(0))
    # cons[7] = L3 (L<=0): -2X4-0.8X5-0.8X6+2X7+X8+2X9 <= 0
    ExaModels.@add_con(c, -2 * x[4] - T(0.8) * x[5] - T(0.8) * x[6] + 2 * x[7] + x[8] + 2 * x[9]; lcon = T(-Inf), ucon = T(0))
    # cons[8] = L4 (L<=0): -0.8X5-0.8X6+X8 <= 0
    ExaModels.@add_con(c, -T(0.8) * x[5] - T(0.8) * x[6] + x[8]; lcon = T(-Inf), ucon = T(0))
    # cons[9] = L5 (L<=0): -X4+X7+X9 <= 0
    ExaModels.@add_con(c, -x[4] + x[7] + x[9]; lcon = T(-Inf), ucon = T(0))
    # cons[10] = L6 (L<=0): -0.4X5-0.4X6+1.5X8 <= 0
    ExaModels.@add_con(c, -T(0.4) * x[5] - T(0.4) * x[6] + T(1.5) * x[8]; lcon = T(-Inf), ucon = T(0))
    # cons[11] = L7 (L<=0): 0.16X5+0.16X6-1.2X8 <= 0
    ExaModels.@add_con(c, T(0.16) * x[5] + T(0.16) * x[6] - T(1.2) * x[8]; lcon = T(-Inf), ucon = T(0))
    # cons[12] = L8 (L<=0): X3-0.8X4 <= 0
    ExaModels.@add_con(c, x[3] - T(0.8) * x[4]; lcon = T(-Inf), ucon = T(0))
    # cons[13] = L9 (L<=0): -X3+0.4X4 <= 0
    ExaModels.@add_con(c, -x[3] + T(0.4) * x[4]; lcon = T(-Inf), ucon = T(0))
    # cons[14] = L10 (L<=0): X7-10Y3 <= 0
    ExaModels.@add_con(c, x[7] - 10 * x[12]; lcon = T(-Inf), ucon = T(0))
    # cons[15] = L11 (L<=0): 0.8X5+0.8X6-10Y4 <= 0
    ExaModels.@add_con(c, T(0.8) * x[5] + T(0.8) * x[6] - 10 * x[13]; lcon = T(-Inf), ucon = T(0))
    # cons[16] = L12 (L<=0): 2X4-2X7-2X9-10Y5 <= 0
    ExaModels.@add_con(c, 2 * x[4] - 2 * x[7] - 2 * x[9] - 10 * x[14]; lcon = T(-Inf), ucon = T(0))
    # cons[17] = L13 (L<=0): X5-10Y6 <= 0
    ExaModels.@add_con(c, x[5] - 10 * x[15]; lcon = T(-Inf), ucon = T(0))
    # cons[18] = L14 (L<=0): X6-10Y7 <= 0
    ExaModels.@add_con(c, x[6] - 10 * x[16]; lcon = T(-Inf), ucon = T(0))
    # cons[19] = L15 (L<=0): X3+X4-10Y8 <= 0
    ExaModels.@add_con(c, x[3] + x[4] - 10 * x[17]; lcon = T(-Inf), ucon = T(0))
    # cons[20] = N4 (L<=0): exp(X2/1.2)-10*Y2-1 <= 0
    ExaModels.@add_con(c, exp(x[2] / T(1.2)) - 10 * x[11] - 1; lcon = T(-Inf), ucon = T(0))
    # cons[21] = N3 (L<=0): exp(X1)-10*Y1-1 <= 0
    ExaModels.@add_con(c, exp(x[1]) - 10 * x[10] - 1; lcon = T(-Inf), ucon = T(0))
    # cons[22] = N2 (L<=0): -log(X3+X4+1) <= 0
    ExaModels.@add_con(c, -log(x[3] + x[4] + 1); lcon = T(-Inf), ucon = T(0))
    # cons[23] = N1 (L<=0): -X8-1.5*log(X5+1)-log(X6+1) <= 0
    ExaModels.@add_con(c, -x[8] - T(1.5) * log(x[5] + 1) - log(x[6] + 1); lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
