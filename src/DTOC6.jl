# DTOC6: Discrete-time optimal control problem 6
# N=5001 periods, 1 control, 1 state variable; all free
# Groups OY(T) and OX(T) for T=1..N-1 with SCALE=2, L2 group type
# OY(T) uses element E(T) (EXP type: exp(X(T)))
# TT(T) constraint also uses E(T)
# Objective: sum_{T=1..N-1} [(Y(T))^2/2 + exp(X(T))]  (from L2 scale=2 and exp element)
# Actually: OY(T) has SCALE=2, L2 type, element E(T): contribution = (Y(T) + exp(X(T)))^2 / 2
# OX(T) has SCALE=2, L2 type, linear X(T): contribution = X(T)^2 / 2
# TT(T): -Y(T+1) + Y(T) + E(T) = 0 → -Y(T+1) + Y(T) + exp(X(T)) = 0
# Fixed: Y(1) = 0
function dtoc6_model(N = 5001; T = Float64, backend = nothing, kwargs...)
    NV = (N - 1) + N  # = 2N-1
    x_idx(t) = t           # T=1..N-1
    y_idx(t) = (N - 1) + t  # T=1..N

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    lvar[y_idx(1)] = T(0.0); uvar[y_idx(1)] = T(0.0)

    x0 = zeros(T, NV)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective:
    # OY(T): L2 group type, SCALE=2, linear Y(T) + element exp(X(T))
    #   contribution = (Y(T) + exp(X(T)))^2 / 2
    # OX(T): L2 group type, SCALE=2, linear X(T)
    #   contribution = X(T)^2 / 2
    odata = [(yt = y_idx(t), xt = x_idx(t)) for t in 1:(N-1)]
    ExaModels.@add_obj(c, (x[d.yt] + exp(x[d.xt]))^2 / T(2) for d in odata)
    ExaModels.@add_obj(c, x[d.xt]^2 / T(2) for d in odata)

    # Constraints TT(T): -Y(T+1) + Y(T) + exp(X(T)) = 0
    cdata = [(ytp1 = y_idx(t + 1), yt = y_idx(t), xt = x_idx(t)) for t in 1:(N-1)]
    ExaModels.@add_con(c, -x[d.ytp1] + x[d.yt] + exp(x[d.xt]) for d in cdata)

    return ExaModels.ExaModel(c; kwargs...)
end
