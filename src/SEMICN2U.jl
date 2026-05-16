# SEMICN2U: Unconstrained version of SEMICON2 (N=5000 default, LN=4500)
# Same as SEMICON2 but bounds are FR (free) instead of box constraints
# LAMBDA=0.2, all variables free except boundary (U(0)=LUA, U(N+1)=LUB fixed)
function semicn2u_model(N = 5000, LN = 4500; T = Float64, backend = nothing, kwargs...)
    A = T(-0.00009)
    B = T(0.00001)
    UA = T(0.0)
    UB = T(700.0)
    CA = T(1.0e12)
    CB = T(1.0e13)
    BETA = T(40.0)
    LAMBDA = T(0.2)

    H = (B - A) / (N + 1)
    H2 = H^2
    LB  = LAMBDA * BETA
    LH2CA = LAMBDA * H2 * CA
    LH2CB = LAMBDA * H2 * CB
    LUA = LAMBDA * UA
    LUB = LAMBDA * UB

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    lvar = fill(T(-Inf), N + 2)
    uvar = fill(T(Inf), N + 2)
    lvar[1]     = LUA;   uvar[1]     = LUA
    lvar[N + 2] = LUB;   uvar[N + 2] = LUB
    start = fill(T(0), N + 2)
    start[1]     = LUA
    start[N + 2] = LUB
    ExaModels.@add_var(c, x, N + 2; lvar = lvar, uvar = uvar, start = start)

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
