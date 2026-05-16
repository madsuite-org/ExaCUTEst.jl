# ERRINBAR: Nonlinearly constrained structural problem, 18 variables
# Variables: U(1..8) at positions 1-8, X(1..10) at positions 9-18
# Objective: linear combination of X(I) with coefficients C0 or C0*sqrt(2)
# Constraints: 8 equality + 1 inequality (structural force balance)
# U(I) free (FR default), X(I) >= 0.645; U(2),U(4),U(6),U(8) >= -50.8
function errinbar_model(; T = Float64, backend = nothing, kwargs...)
    NV = 18
    SQ2 = sqrt(T(2))
    SQ8 = sqrt(T(8))
    inv8 = T(1) / SQ8
    C0    = T(2.53106)
    C0SQ2 = C0 * SQ2
    P0    = T(589.884)  # -P0 constant in SIF = 589.884 as RHS

    # Variable indices: U(i) = i, X(i) = 8+i
    U(i) = i
    X(i) = 8 + i

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    # X(I) >= 0.645
    for i in 1:10
        lvar[X(i)] = T(0.645)
    end
    # U(2),U(4),U(6),U(8) >= -50.8
    for i in [2, 4, 6, 8]
        lvar[U(i)] = T(-50.8)
    end

    x0 = zeros(T, NV)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: linear in X(I) (N-type identity group)
    ExaModels.@add_obj(c,
        C0 * x[X(1)] + C0SQ2 * x[X(2)] + C0SQ2 * x[X(3)] + C0 * x[X(4)] +
        C0 * x[X(5)] + C0 * x[X(6)] + C0SQ2 * x[X(7)] + C0SQ2 * x[X(8)] +
        C0 * x[X(9)] + C0 * x[X(10)])

    # Element shortcuts (expanded in-place):
    # EA = X1*U1 = x[X(1)]*x[U(1)]
    # EB = XUPV(X3,U1,U2) = X3*(U1+U2) = x[X(3)]*(x[U(1)]+x[U(2)])
    # EC = XUMV(X6,U1,U5) = X6*(U1-U5) = x[X(6)]*(x[U(1)]-x[U(5)])
    # ED = XBIG(X7,U1,U8,U2,U7) = X7*(U1+U8-U2-U7)
    # EE = XUMV(X5,U2,U4) = X5*(U2-U4)
    # EF = XUMV(X2,U3,U4) = X2*(U3-U4)
    # EG = XU(X4,U3) = X4*U3
    # EH = XBIG(X8,U3,U4,U5,U6) = X8*(U3+U4-U5-U6)
    # EI = XUMV(X9,U3,U7) = X9*(U3-U7)
    # EJ = XUMV(X10,U6,U8) = X10*(U6-U8)

    # C1 = EA + EC + (1/SQ8)*EB + (1/SQ8)*ED = 0
    ExaModels.@add_con(c,
        x[X(1)] * x[U(1)] +
        x[X(6)] * (x[U(1)] - x[U(5)]) +
        inv8 * x[X(3)] * (x[U(1)] + x[U(2)]) +
        inv8 * x[X(7)] * (x[U(1)] + x[U(8)] - x[U(2)] - x[U(7)]))

    # C2 = EB + EE + (1/SQ8)*ED = 0
    ExaModels.@add_con(c,
        x[X(3)] * (x[U(1)] + x[U(2)]) +
        x[X(5)] * (x[U(2)] - x[U(4)]) +
        inv8 * x[X(7)] * (x[U(1)] + x[U(8)] - x[U(2)] - x[U(7)]))

    # C3 = (1/SQ8)*EF + (1/SQ8)*EG + (1/SQ8)*EH + EI = 0
    ExaModels.@add_con(c,
        inv8 * x[X(2)] * (x[U(3)] - x[U(4)]) +
        inv8 * x[X(4)] * x[U(3)] +
        inv8 * x[X(8)] * (x[U(3)] + x[U(4)] - x[U(5)] - x[U(6)]) +
        x[X(9)] * (x[U(3)] - x[U(7)]))

    # C4 = (1/SQ8)*EF + (1/SQ8)*EH + EE - rhs=0 where rhs=-P0=-589.884 => C4=P0=589.884
    # Wait: SIF says Z ERRINBAR C4 -P0, -P0 = 589.884, so C4 group constant=589.884 (RHS=589.884)
    # constraint value = element_sum - 589.884 = 0
    ExaModels.@add_con(c,
        inv8 * x[X(2)] * (x[U(3)] - x[U(4)]) +
        inv8 * x[X(8)] * (x[U(3)] + x[U(4)] - x[U(5)] - x[U(6)]) +
        x[X(5)] * (x[U(2)] - x[U(4)]) - P0)

    # C5 = (1/SQ8)*EH + EC = 0
    ExaModels.@add_con(c,
        inv8 * x[X(8)] * (x[U(3)] + x[U(4)] - x[U(5)] - x[U(6)]) +
        x[X(6)] * (x[U(1)] - x[U(5)]))

    # C6 = (1/SQ8)*EH + EJ = 0
    ExaModels.@add_con(c,
        inv8 * x[X(8)] * (x[U(3)] + x[U(4)] - x[U(5)] - x[U(6)]) +
        x[X(10)] * (x[U(6)] - x[U(8)]))

    # C7 = (1/SQ8)*ED + EI = 0
    ExaModels.@add_con(c,
        inv8 * x[X(7)] * (x[U(1)] + x[U(8)] - x[U(2)] - x[U(7)]) +
        x[X(9)] * (x[U(3)] - x[U(7)]))

    # C8 = (1/SQ8)*ED + EJ - rhs=-P0=589.884
    ExaModels.@add_con(c,
        inv8 * x[X(7)] * (x[U(1)] + x[U(8)] - x[U(2)] - x[U(7)]) +
        x[X(10)] * (x[U(6)] - x[U(8)]) - P0)

    # STR: U2 - U4 <= 22.86 (L-type, linear, rhs=22.86)
    ExaModels.@add_con(c,
        x[U(2)] - x[U(4)] - T(22.86);
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
