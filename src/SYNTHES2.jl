# SYNTHES2: Synthesis of processing system 2 (MINLP treated as NLP)
# Source: Duran & Grossmann, Math. Prog. 36 (1986), pp. 307-339, problem 2
# 11 vars: X1..X6 continuous, Y1..Y5 INTEGER→continuous
# CUTEst variable ordering: X1,X2,X5,X4,X3,X6,Y1,Y2,Y3,Y4,Y5
# (determined by sifdecoder variable reordering)
# CUTEst constraint ordering: L10,L11,L9,L1,L2,L3,L4,L5,L6,L7,L8,N3,N2,N1
function synthes2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # x[1]=X1(ub=2), x[2]=X2(ub=2), x[3]=X5(ub=∞), x[4]=X4(ub=∞),
    # x[5]=X3(ub=2), x[6]=X6(ub=3), x[7..11]=Y1..Y5(ub=1)
    ExaModels.@add_var(c, x, 11;
        lvar = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        uvar = T[2.0, 2.0, Inf, Inf, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # Objective (constant from CONSTANTS: -140 → group_value = linear + 140)
    ExaModels.@add_obj(c, 5 * x[7] + 8 * x[8] + 6 * x[9] + 10 * x[10] + 6 * x[11] -
                          10 * x[1] - 15 * x[2] - 15 * x[5] + 15 * x[4] + 5 * x[3] - 20 * x[6] + 140 +
                          exp(x[1]) + exp(x[2] / T(1.2)) - 60 * log(x[4] + x[3] + 1))
    # Constraint ordering from CUTEst sifdecoder:
    # L10 (E): Y1+Y2-1 = 0
    ExaModels.@add_con(c, x[7] + x[8] - 1)
    # L11 (L with RANGE=-1): -1 ≤ Y4+Y5-1 ≤ 0
    ExaModels.@add_con(c, x[10] + x[11] - 1; lcon = T(-1), ucon = T(0))
    # L9 (L): 0.2*X4-X5 ≤ 0
    ExaModels.@add_con(c, T(0.2) * x[4] - x[3]; lcon = T(-Inf), ucon = T(0))
    # L1 (L): 1.25*X3-10*Y3 ≤ 0
    ExaModels.@add_con(c, T(1.25) * x[5] - 10 * x[9]; lcon = T(-Inf), ucon = T(0))
    # L2 (L): X4+X5-10*Y4 ≤ 0
    ExaModels.@add_con(c, x[4] + x[3] - 10 * x[10]; lcon = T(-Inf), ucon = T(0))
    # L3 (L): -2*X3+2*X6-10*Y5 ≤ 0
    ExaModels.@add_con(c, -2 * x[5] + 2 * x[6] - 10 * x[11]; lcon = T(-Inf), ucon = T(0))
    # L4 (L): -X1-X2-2*X3+X4+2*X6 ≤ 0
    ExaModels.@add_con(c, -x[1] - x[2] - 2 * x[5] + x[4] + 2 * x[6]; lcon = T(-Inf), ucon = T(0))
    # L5 (L): -X1-X2-0.75*X3+X4+2*X6 ≤ 0
    ExaModels.@add_con(c, -x[1] - x[2] - T(0.75) * x[5] + x[4] + 2 * x[6]; lcon = T(-Inf), ucon = T(0))
    # L6 (L): X3-X6 ≤ 0
    ExaModels.@add_con(c, x[5] - x[6]; lcon = T(-Inf), ucon = T(0))
    # L7 (L): 2*X3-X4-2*X6 ≤ 0
    ExaModels.@add_con(c, 2 * x[5] - x[4] - 2 * x[6]; lcon = T(-Inf), ucon = T(0))
    # L8 (L): -0.5*X4+X5 ≤ 0
    ExaModels.@add_con(c, -T(0.5) * x[4] + x[3]; lcon = T(-Inf), ucon = T(0))
    # N3 (L): exp(X2/1.2)-10*Y2-1 ≤ 0
    ExaModels.@add_con(c, exp(x[2] / T(1.2)) - 10 * x[8] - 1; lcon = T(-Inf), ucon = T(0))
    # N2 (L): exp(X1)-10*Y1-1 ≤ 0
    ExaModels.@add_con(c, exp(x[1]) - 10 * x[7] - 1; lcon = T(-Inf), ucon = T(0))
    # N1 (L): -log(X4+X5+1) ≤ 0
    ExaModels.@add_con(c, -log(x[4] + x[3] + 1); lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
