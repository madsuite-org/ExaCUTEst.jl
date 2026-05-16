# MCONCON: gas network problem, 15 variables, 11 constraints
# Variables: P(1..7), Q(1..4), F(1..4)
# Variable ordering (SIF): P1,Q1,F1,P2,Q2,F2,P3,Q3,F3,P4,Q4,F4,P5,P6,P7
# (DO I 1..M: X P(I), X Q(I), X F(I); then DO I M+1..N: X P(I))
# Objective: minimize -sum P(i) for i=1..7
# PAN(I) constraints: P1^2|X|P2^2 + K*Q1^|Q1|^0.8539 = 0 (nonlinear)
# MBAL(I): linear flow balance
# Bounds: all free by default
# K = -0.597053452
# SQR element: X*|X| (F = X*abs(X))
# FORQ element: Y*|Y|^0.8539 (F = Y*abs(Y)^0.8539)
# PAN1: P1^2 - P2^2 + K*Q1*|Q1|^0.8539 = 0 → P(1)^2-P(2)^2+K*Q1*|Q1|^0.8539=0
# But from SIF: PSQ1=P1^2 sign, PSQ2=-P2^2 sign → P1*|P1| - P2*|P2| + K*Q1*|Q1|^0.8539=0
# Actually SQR element: F=X*abs(X), so PSQ(I) = P(I)*abs(P(I))
# PAN1: PSQ1 coeff 1 + PSQ2 coeff -1 + QTO1 coeff K
#   so: P1*|P1| - P2*|P2| + K*Q1*|Q1|^0.8539 = 0
# PAN2: PSQ3 + (-PSQ4) + K*QTO2 → P3*|P3| - P4*|P4| + K*Q2*|Q2|^0.8539 = 0
# PAN3: PSQ4 + (-PSQ5) + K*QTO3 → P4*|P4| - P5*|P5| + K*Q3*|Q3|^0.8539 = 0
# PAN4: PSQ6 + (-PSQ7) + K*QTO4 → P6*|P6| - P7*|P7| + K*Q4*|Q4|^0.8539 = 0
# Linear constraints MBAL:
# MBAL1: Q1 - F3 = 0
# MBAL2: -Q1 + F1 = 0
# MBAL3: Q2 - F1 = 0
# MBAL4: -Q2 + Q3 = -1000 (rhs from CONSTANTS: DEMAND=-1000)
# MBAL5: -Q3 - F2 = 0
# MBAL6: Q4 + F2 = 0
# MBAL7: -Q4 - F4 = 0
# Bounds: FR default, P3<=904.73, P5<=904.73, P1<=914.73, P7<=914.73, F4<=400
function mconcon_model(; T = Float64, backend = nothing, kwargs...)
    K = T(-0.597053452)
    # Variable order: x[1..15] = P1,Q1,F1,P2,Q2,F2,P3,Q3,F3,P4,Q4,F4,P5,P6,P7
    # p1=x[1],q1=x[2],f1=x[3],p2=x[4],q2=x[5],f2=x[6],p3=x[7],q3=x[8],f3=x[9]
    # p4=x[10],q4=x[11],f4=x[12],p5=x[13],p6=x[14],p7=x[15]
    lvar = fill(T(-Inf), 15)
    uvar = fill(T(Inf), 15)
    uvar[1] = T(914.73)   # P1 <= 914.73
    uvar[7] = T(904.73)   # P3 <= 904.73
    uvar[13] = T(904.73)  # P5 <= 904.73
    uvar[15] = T(914.73)  # P7 <= 914.73
    uvar[12] = T(400.0)   # F4 <= 400
    start = fill(T(965.0), 15)
    start[2] = T(100.0)  # Q1
    start[5] = T(100.0)  # Q2
    start[8] = T(-100.0) # Q3
    start[11] = T(-100.0)# Q4
    start[3] = T(1000.0) # F1
    start[6] = T(1000.0) # F2
    start[9] = T(1000.0) # F3
    start[12] = T(1000.0)# F4

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 15; lvar = lvar, uvar = uvar, start = start)

    # Objective: minimize -sum P(i) for i=1..7
    ExaModels.@add_obj(c, -(x[1] + x[4] + x[7] + x[10] + x[13] + x[14] + x[15]))

    # PAN1: P1*|P1| - P2*|P2| + K*Q1*|Q1|^0.8539 = 0
    ExaModels.@add_con(c, x[1] * abs(x[1]) - x[4] * abs(x[4]) + K * x[2] * abs(x[2])^T(0.8539))
    # PAN2: P3*|P3| - P4*|P4| + K*Q2*|Q2|^0.8539 = 0
    ExaModels.@add_con(c, x[7] * abs(x[7]) - x[10] * abs(x[10]) + K * x[5] * abs(x[5])^T(0.8539))
    # PAN3: P4*|P4| - P5*|P5| + K*Q3*|Q3|^0.8539 = 0
    ExaModels.@add_con(c, x[10] * abs(x[10]) - x[13] * abs(x[13]) + K * x[8] * abs(x[8])^T(0.8539))
    # PAN4: P6*|P6| - P7*|P7| + K*Q4*|Q4|^0.8539 = 0
    ExaModels.@add_con(c, x[14] * abs(x[14]) - x[15] * abs(x[15]) + K * x[11] * abs(x[11])^T(0.8539))
    # MBAL1: Q1 - F3 = 0
    ExaModels.@add_con(c, x[2] - x[9])
    # MBAL2: -Q1 + F1 = 0
    ExaModels.@add_con(c, -x[2] + x[3])
    # MBAL3: Q2 - F1 = 0
    ExaModels.@add_con(c, x[5] - x[3])
    # MBAL4: -Q2 + Q3 = -1000
    ExaModels.@add_con(c, -x[5] + x[8] + T(1000))
    # MBAL5: -Q3 - F2 = 0
    ExaModels.@add_con(c, -x[8] - x[6])
    # MBAL6: Q4 + F2 = 0
    ExaModels.@add_con(c, x[11] + x[6])
    # MBAL7: -Q4 - F4 = 0
    ExaModels.@add_con(c, -x[11] - x[12])

    return ExaModels.ExaModel(c; kwargs...)
end
