# SCOND1LS: Semiconductor problem, least-squares version of SEMICON1 (N=5000 default)
# Same structure as SEMICON1 but groups are XN (objective) with L2 group type
# instead of equality constraints
function scond1ls_model(N = 5000, LN = 4500; T = Float64, backend = nothing, kwargs...)
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
    LUA = LAMBDA * UA
    LUB = LAMBDA * UB
    ULW = LUA - T(5)
    UUP = LUB + T(5)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    lvar = fill(ULW, N + 2)
    uvar = fill(UUP, N + 2)
    lvar[1]     = LUA;   uvar[1]     = LUA
    lvar[N + 2] = LUB;   uvar[N + 2] = LUB
    start = fill(T(0), N + 2)
    start[1]     = LUA
    start[N + 2] = LUB
    ExaModels.@add_var(c, x, N + 2; lvar = lvar, uvar = uvar, start = start)

    # Objective: sum of (G(i))^2 via L2 group type
    # G(i) value = U(i-1) - 2*U(i) + U(i+1) + EA(i) + EB(i) - rhs(i)
    data_neg = [(i = i,) for i in 1:LN]
    data_pos = [(i = i,) for i in LN+1:N]

    ExaModels.@add_obj(c,
        (x[d.i] - T(2) * x[d.i + 1] + x[d.i + 2] +
         LH2CA * exp(-LB * (x[d.i + 1] - LUA)) +
         (-LH2CB) * exp(LB * (x[d.i + 1] - LUB)) - LH2CA)^2
        for d in data_neg)
    ExaModels.@add_obj(c,
        (x[d.i] - T(2) * x[d.i + 1] + x[d.i + 2] +
         LH2CA * exp(-LB * (x[d.i + 1] - LUA)) +
         (-LH2CB) * exp(LB * (x[d.i + 1] - LUB)) - (-LH2CB))^2
        for d in data_pos)

    return ExaModels.ExaModel(c; kwargs...)
end
