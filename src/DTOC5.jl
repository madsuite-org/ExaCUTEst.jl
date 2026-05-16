# DTOC5: Discrete-time optimal control problem 5 (convex QP)
# N=5000 periods, 1 control, 1 state variable
# H = 1/N (S in SIF = 1/N)
# OBJ group SCALE = N (RN), identity group type
# GROUP USES: OBJ uses YSQ(T) coeff 1, XSQ(T) coeff 1 for T=1..N-1
#             TT(T) uses YSQ(T) coeff H (SIF: ZE TT(T) YSQ(T) H)
# So objective contribution = (1/N) * sum_{T=1..N-1} (Y(T)^2 + X(T)^2)
# Constraints TT(T): -Y(T+1) + Y(T) - H*X(T) + H*Y(T)^2 = 0
# Fixed: Y(1) = 1
function dtoc5_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    H  = T(1.0 / N)
    RN = T(N)
    invN = T(1.0 / N)

    NV = (N - 1) + N  # = 2N-1
    x_idx(t) = t         # T=1..N-1
    y_idx(t) = (N - 1) + t  # T=1..N

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    lvar[y_idx(1)] = T(1.0); uvar[y_idx(1)] = T(1.0)

    x0 = zeros(T, NV)
    x0[y_idx(1)] = T(1.0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # OBJ: identity group, SCALE = N, contribution = v/N
    # v = sum_{T=1..N-1} (Y(T)^2 + X(T)^2)
    odata = [(yt = y_idx(t), xt = x_idx(t)) for t in 1:(N-1)]
    ExaModels.@add_obj(c, invN * (x[d.yt]^2 + x[d.xt]^2) for d in odata)

    # Constraints TT(T): -Y(T+1) + Y(T) - H*X(T) + H*Y(T)^2 = 0
    cdata = [(ytp1 = y_idx(t + 1), yt = y_idx(t), xt = x_idx(t)) for t in 1:(N-1)]
    ExaModels.@add_con(c, -x[d.ytp1] + x[d.yt] - H * x[d.xt] + H * x[d.yt]^2 for d in cdata)

    return ExaModels.ExaModel(c; kwargs...)
end
