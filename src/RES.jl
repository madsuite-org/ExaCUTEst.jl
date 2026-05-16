# RES: Dassault spring design problem (20 variables, 14 constraints)
# Source: A.R. Conn, June 1993.
# classification NOR2-MN-20-14
# Variables: L0, N, F, K, LB, L, DE, DI, TO, TOB, NU, D, P, E, P0, G, DM, FR(var), TOLIM, TOBLIM
# Constraints E1-E12 are equality, E13-E14 are inequality
# Elements EL1-EL5 are defined but never used in GROUP USES → purely linear problem
function res_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # Variables: indices 1..20
    # L0(1), N(2), F(3), K(4), LB(5), L(6), DE(7), DI(8), TO(9), TOB(10),
    # NU(11), D(12), P(13), E(14), P0(15), G(16), DM(17), FR(18), TOLIM(19), TOBLIM(20)
    ExaModels.@add_var(c, x, 20;
        lvar = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.5, 0.1, 0.0, 0.0, 1.0, 40000.0, 0.1, 0.0, 100.0, 100.0],
        uvar = T[100.0, 100.0, 30.0, 100.0, 50.0, 50.0, 30.0, 30.0, 800.0, 800.0,
                 50.0, 10.0, 20.0, 10.0, 1000.0, 80000.0, 30.0, 50.0, 1000.0, 1000.0],
        start = T[0.15, 24.079, 9.2459e-15, 0.0, 0.0, 0.15, 6.812, 6.612, 0.0, 0.0,
                  22.079, 0.1, 0.65268, 0.55268, 658.87, 65887.0, 6.712, 0.15, 100.0, 100.0])
    # E1: -F = 0  (x[3] = F)
    ExaModels.@add_con(c, -x[3])
    # E2: -K = 0  (x[4] = K)
    ExaModels.@add_con(c, -x[4])
    # E3: DE - D - DM = 0  (x[7]-x[12]-x[17])
    ExaModels.@add_con(c, x[7] - x[12] - x[17])
    # E4: DI + D - DM = 0  (x[8]+x[12]-x[17])
    ExaModels.@add_con(c, x[8] + x[12] - x[17])
    # E5: D - P + E = 0  (x[12]-x[13]+x[14])
    ExaModels.@add_con(c, x[12] - x[13] + x[14])
    # E6: NU - N = -2  → NU - N + 2 = 0  (x[11]-x[2]+2)
    ExaModels.@add_con(c, x[11] - x[2] + T(2.0))
    # E7: 1.5*D - L0 = 0  (1.5*x[12]-x[1])
    ExaModels.@add_con(c, T(1.5) * x[12] - x[1])
    # E8: L - LB - FR = 0  (x[6]-x[5]-x[18], FR is variable x[18])
    ExaModels.@add_con(c, x[6] - x[5] - x[18])
    # E9: LB = 0  (x[5])
    ExaModels.@add_con(c, x[5])
    # E10: L - L0 + F = 0  (x[6]-x[1]+x[3])
    ExaModels.@add_con(c, x[6] - x[1] + x[3])
    # E11: TO = 0  (x[9])
    ExaModels.@add_con(c, x[9])
    # E12: TOB = 0  (x[10])
    ExaModels.@add_con(c, x[10])
    # E13: TO - TOLIM ≤ 0  (lcon=-Inf, ucon=0)
    ExaModels.@add_con(c, x[9] - x[19]; lcon = T(-Inf), ucon = T(0.0))
    # E14: TOB - TOBLIM ≤ 0  (lcon=-Inf, ucon=0)
    ExaModels.@add_con(c, x[10] - x[20]; lcon = T(-Inf), ucon = T(0.0))
    return ExaModels.ExaModel(c; kwargs...)
end
