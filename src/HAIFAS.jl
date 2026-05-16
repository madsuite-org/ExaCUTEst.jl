# HAIFAS: Truss Topology Design (t6-9), 13 variables, 9 constraints
# Variables: Z (x[1]) and X(1)..X(12) (x[2]..x[13]), all free (XR)
# Objective: minimize Z = x[1]
# Constraints G(J) for J=1..9 (L type, lcon=-Inf, ucon=0):
#   group value = -Z - X(10) + sum_L coeff_L * E(L) <= 0
#   i.e., -x[1] - x[11] + [elements] <= 0
# 2PR element: F = 0.5 * X * Y (symmetric bilinear)
#
# Variable mapping: Z→x[1], X(I)→x[I+1] for I=1..12
# So X(4)→x[5], X(5)→x[6], X(6)→x[7], X(10)→x[11], X(11)→x[12], X(12)→x[13]
#
# Element E(L) = 0.5 * X(I1) * X(I2) = 0.5 * x[I1+1] * x[I2+1]
# Elements (from SIF):
# L=1: (4,4): 0.5*x[5]^2     L=2: (5,5): 0.5*x[6]^2    L=3: (5,11): 0.5*x[6]*x[12]
# L=4: (11,11): 0.5*x[12]^2  L=5: (10,10): 0.5*x[11]^2  L=6: (10,11): 0.5*x[11]*x[12]
# L=7: (11,11): 0.5*x[12]^2  L=8: (4,4): 0.5*x[5]^2     L=9: (4,10): 0.5*x[5]*x[11]
# L=10: (10,10): 0.5*x[11]^2 L=11: (5,5): 0.5*x[6]^2    L=12: (6,6): 0.5*x[7]^2
# L=13: (6,12): 0.5*x[7]*x[13] L=14: (12,12): 0.5*x[13]^2 L=15: (11,11): 0.5*x[12]^2
# L=16: (11,12): 0.5*x[12]*x[13] L=17: (12,12): 0.5*x[13]^2 L=18: (5,5): 0.5*x[6]^2
# L=19: (5,11): 0.5*x[6]*x[12] L=20: (11,11): 0.5*x[12]^2 L=21: (6,6): 0.5*x[7]^2
#
# GROUP USES for G(J):
# G(1): E(1)*10 → 10*0.5*x[5]^2 = 5*x[5]^2
# G(2): E(2)*6.4 + E(3)*6.4 + E(4)*1.6 → 3.2*x[6]^2 + 3.2*x[6]*x[12] + 0.8*x[12]^2
# G(3): E(5)*40 + E(6)*(-80) + E(7)*40 → 20*x[11]^2 - 40*x[11]*x[12] + 20*x[12]^2 = 20*(x[11]-x[12])^2
# G(4): E(8)*6.4 + E(9)*(-6.4) + E(10)*1.6 → 3.2*x[5]^2 - 3.2*x[5]*x[11] + 0.8*x[11]^2
# G(5): E(11)*10 → 5*x[6]^2
# G(6): E(12)*6.4 + E(13)*6.4 + E(14)*1.6 → 3.2*x[7]^2 + 3.2*x[7]*x[13] + 0.8*x[13]^2
# G(7): E(15)*40 + E(16)*(-80) + E(17)*40 → 20*x[12]^2 - 40*x[12]*x[13] + 20*x[13]^2 = 20*(x[12]-x[13])^2
# G(8): E(18)*6.4 + E(19)*(-6.4) + E(20)*1.6 → 3.2*x[6]^2 - 3.2*x[6]*x[12] + 0.8*x[12]^2
# G(9): E(21)*10 → 5*x[7]^2
#
# Constraint residual = group_value - 0 (no constants), L type: value <= 0
# CUTEst convention: residual = group_value, lcon=-Inf, ucon=0
# In ExaModels: @add_con(c, expr, lcon=-Inf, ucon=0)
# where expr = -Z - X(10) + element_contributions = -x[1] - x[11] + elements
function haifas_model(; T = Float64, backend = nothing, kwargs...)
    # 13 variables: x[1]=Z, x[2..13]=X(1..12)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 13; lvar = T(-Inf), uvar = T(Inf))

    # Minimize Z = x[1]
    ExaModels.@add_obj(c, x[1])

    # G(1): -x[1] - x[11] + 5*x[5]^2 <= 0
    ExaModels.@add_con(c, -x[1] - x[11] + T(5) * x[5]^2; lcon = T(-Inf), ucon = T(0))

    # G(2): -x[1] - x[11] + 3.2*x[6]^2 + 3.2*x[6]*x[12] + 0.8*x[12]^2 <= 0
    ExaModels.@add_con(c, -x[1] - x[11] + T(3.2) * x[6]^2 + T(3.2) * x[6] * x[12] + T(0.8) * x[12]^2;
                       lcon = T(-Inf), ucon = T(0))

    # G(3): -x[1] - x[11] + 20*(x[11]-x[12])^2 <= 0
    ExaModels.@add_con(c, -x[1] - x[11] + T(20) * (x[11] - x[12])^2; lcon = T(-Inf), ucon = T(0))

    # G(4): -x[1] - x[11] + 3.2*x[5]^2 - 3.2*x[5]*x[11] + 0.8*x[11]^2 <= 0
    ExaModels.@add_con(c, -x[1] - x[11] + T(3.2) * x[5]^2 - T(3.2) * x[5] * x[11] + T(0.8) * x[11]^2;
                       lcon = T(-Inf), ucon = T(0))

    # G(5): -x[1] - x[11] + 5*x[6]^2 <= 0
    ExaModels.@add_con(c, -x[1] - x[11] + T(5) * x[6]^2; lcon = T(-Inf), ucon = T(0))

    # G(6): -x[1] - x[11] + 3.2*x[7]^2 + 3.2*x[7]*x[13] + 0.8*x[13]^2 <= 0
    ExaModels.@add_con(c, -x[1] - x[11] + T(3.2) * x[7]^2 + T(3.2) * x[7] * x[13] + T(0.8) * x[13]^2;
                       lcon = T(-Inf), ucon = T(0))

    # G(7): -x[1] - x[11] + 20*(x[12]-x[13])^2 <= 0
    ExaModels.@add_con(c, -x[1] - x[11] + T(20) * (x[12] - x[13])^2; lcon = T(-Inf), ucon = T(0))

    # G(8): -x[1] - x[11] + 3.2*x[6]^2 - 3.2*x[6]*x[12] + 0.8*x[12]^2 <= 0
    ExaModels.@add_con(c, -x[1] - x[11] + T(3.2) * x[6]^2 - T(3.2) * x[6] * x[12] + T(0.8) * x[12]^2;
                       lcon = T(-Inf), ucon = T(0))

    # G(9): -x[1] - x[11] + 5*x[7]^2 <= 0
    ExaModels.@add_con(c, -x[1] - x[11] + T(5) * x[7]^2; lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
