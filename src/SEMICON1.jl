# SEMICON1: Semiconductor problem (N=5000 default, LN=4500)
# Source: More' (1988), problem 10
# N+2 variables U(0)..U(N+1); U(0) and U(N+1) fixed
# N equality constraints G(i) for i=1..N:
#   U(i-1) - 2*U(i) + U(i+1) + EA(i) + EB(i) - rhs = 0
# where:
#   EA(i) = LH2CA * exp(-LB * (U(i) - LUA))   (for i=1..LN)
#   EB(i) = -LH2CB * exp(LB * (U(i) - LUB))   (for all i)
#   rhs(i) = LH2CA (for i=1..LN), -LH2CB (for i=LN+1..N)
# WE1 element: F = LAC * exp(LAB * (X - LU))
function semicon1_model(N = 5000, LN = 4500; T = Float64, backend = nothing, kwargs...)
    A = T(-0.00009)
    B = T(0.00001)
    UA = T(0.0)
    UB = T(700.0)
    CA = T(1.0e12)
    CB = T(1.0e13)
    BETA = T(40.0)
    LAMBDA = T(1.0)

    H = (B - A) / (N + 1)
    H2 = H^2
    LB  = LAMBDA * BETA
    LH2CA = LAMBDA * H2 * CA
    LH2CB = LAMBDA * H2 * CB
    LUA = LAMBDA * UA    # = 0
    LUB = LAMBDA * UB    # = 700
    ULW = LUA - T(5)
    UUP = LUB + T(5)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # Variables U(0)..U(N+1): indexed 1..N+2 in Julia
    # U(0) = x[1] fixed at LUA=0; U(N+1) = x[N+2] fixed at LUB=700
    lvar = fill(ULW, N + 2)
    uvar = fill(UUP, N + 2)
    lvar[1]     = LUA;   uvar[1]     = LUA
    lvar[N + 2] = LUB;   uvar[N + 2] = LUB
    start = fill(T(0), N + 2)
    start[1]     = LUA
    start[N + 2] = LUB
    ExaModels.@add_var(c, x, N + 2; lvar = lvar, uvar = uvar, start = start)

    # Constraints G(i) for i=1..N (Julia index i -> x[i+1])
    # G(i): U(i-1) - 2*U(i) + U(i+1) + EA(i) + EB(i) - rhs(i) = 0
    # EA(i) = LH2CA * exp(-LB * (x[i+1] - LUA))
    # EB(i) = (-LH2CB) * exp(LB * (x[i+1] - LUB))
    # rhs(i) = LH2CA for i<=LN, -LH2CB for i>LN

    data_neg = [(i = i,) for i in 1:LN]
    data_pos = [(i = i,) for i in LN+1:N]

    ExaModels.@add_con(c,
        x[d.i] - T(2) * x[d.i + 1] + x[d.i + 2] +
        LH2CA * exp(-LB * (x[d.i + 1] - LUA)) +
        (-LH2CB) * exp(LB * (x[d.i + 1] - LUB)) - LH2CA
        for d in data_neg)
    ExaModels.@add_con(c,
        x[d.i] - T(2) * x[d.i + 1] + x[d.i + 2] +
        LH2CA * exp(-LB * (x[d.i + 1] - LUA)) +
        (-LH2CB) * exp(LB * (x[d.i + 1] - LUB)) - (-LH2CB)
        for d in data_pos)

    return ExaModels.ExaModel(c; kwargs...)
end
