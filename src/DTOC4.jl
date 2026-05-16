# DTOC4: Discrete-time optimal control problem 4
# N=1500 periods, 1 control, 2 state variables
# H = 1/N, 5H = 5/N, 1+5H, -5H
# Objective OBJ (single group, SCALE=1/(5H)):
#   OBJ = 0.5*Y1(1)^2 + 0.5*Y2(1)^2 + X(1)^2
#         + sum_{T=2..N-1} (Y1(T)^2 + Y2(T)^2 + X(T)^2)
#         + 0.5*Y1(N)^2 + 0.5*Y2(N)^2
#         - sum_{T=1..N-1} 5H*Y2(T)*Y1(T)
#   Then multiply by 1/(5H) = N/5
# Constraints:
#   TT(T,1): Y(T+1,1) = (1+5H)*Y(T,1) - 5H*Y(T,2) + 5H*X(T)
#   TT(T,2): Y(T+1,2) = Y(T,2) + 5H*Y(T,1)
# Fixed: Y(1,1)=0, Y(1,2)=1
function dtoc4_model(N = 1500; T = Float64, backend = nothing, kwargs...)
    H     = T(1.0 / N)
    fiveH = T(5.0) * H
    inv5H = T(1.0) / fiveH   # = N/5, SCALE = 1/(5H)
    one5H = T(1.0) + fiveH

    NV = (N - 1) + N + N  # = 3N-1
    x_idx(t)  = t            # T=1..N-1
    y1_idx(t) = (N - 1) + t  # T=1..N
    y2_idx(t) = (N - 1) + N + t  # T=1..N

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    lvar[y1_idx(1)] = T(0.0); uvar[y1_idx(1)] = T(0.0)
    lvar[y2_idx(1)] = T(1.0); uvar[y2_idx(1)] = T(1.0)

    x0 = zeros(T, NV)
    x0[y1_idx(1)] = T(0.0)
    x0[y2_idx(1)] = T(1.0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # SIF OBJ group has SCALE = 1/(5H), identity group type
    # GROUP USES: OBJ uses Y1SQ(1) coeff 0.5, Y2SQ(1) coeff 0.5, XSQ(1) coeff 1.0
    #   T=2..N-1: Y1SQ(T) coeff 1, Y2SQ(T) coeff 1, XSQ(T) coeff 1
    #   Y1SQ(N) coeff 0.5, Y2SQ(N) coeff 0.5
    #   TT(T,1) uses E(T) coeff -5H  (AAB element: Y2(T)*Y1(T)^2... wait)
    # Wait: SIF shows ZE TT(T,1) E(T) -5H → E(T) contributes to constraint TT(T,1) NOT to objective
    # Let me re-read: the GROUP USES for OBJ does NOT include E(T)!
    # OBJ = sum of SQ elements = (1/SCALE) * (quadratic part)
    # Actually SIF says: ZN OBJ 'SCALE' 1/5H, and in GROUP USES only SQ elements go to OBJ
    # E(T) (AAB type) only goes to TT(T,1) constraint
    # So OBJ value = (0.5*Y1(1)^2 + 0.5*Y2(1)^2 + X(1)^2 + sum_middle + 0.5*Y1(N)^2 + 0.5*Y2(N)^2) / (1/5H)
    # = (...) * 5H = (...) / (N/5)
    # Wait: identity group with SCALE σ: contribution = v/σ. For objective: obj += v/σ.
    # Here v = sum of SQ elements, SCALE = 1/(5H), so contribution = v * 5H

    # AAB element: A=Y2(T), B=Y1(T), F=A*A*B = Y2(T)^2*Y1(T)
    # ZE TT(T,1) E(T) -5H means TT(T,1) linear part + (-5H)*AAB = 0
    # So constraint TT(T,1): -Y(T+1,1) + (1+5H)*Y(T,1) - 5H*Y(T,2) + 5H*X(T) - 5H*Y2(T)^2*Y1(T) = 0

    # Objective: sum of SQ elements (Y1, Y2, X) scaled by 5H
    # Terminal term:
    ExaModels.@add_obj(c, fiveH * (T(0.5) * x[y1_idx(1)]^2 + T(0.5) * x[y2_idx(1)]^2 + x[x_idx(1)]^2))
    if N > 2
        mid_data = [(y1t = y1_idx(t), y2t = y2_idx(t), xt = x_idx(t)) for t in 2:(N-1)]
        ExaModels.@add_obj(c, fiveH * (x[d.y1t]^2 + x[d.y2t]^2 + x[d.xt]^2) for d in mid_data)
    end
    ExaModels.@add_obj(c, fiveH * (T(0.5) * x[y1_idx(N)]^2 + T(0.5) * x[y2_idx(N)]^2))

    # Constraints TT(T,1): -Y(T+1,1) + (1+5H)*Y(T,1) - 5H*Y(T,2) + 5H*X(T) - 5H*Y2(T)^2*Y1(T) = 0
    # TT(T,2): -Y(T+1,2) + Y(T,2) + 5H*Y(T,1) = 0
    cdata = [(
        y1tp1 = y1_idx(t + 1), y2tp1 = y2_idx(t + 1),
        y1t   = y1_idx(t),     y2t   = y2_idx(t),
        xt    = x_idx(t),
    ) for t in 1:(N-1)]
    ExaModels.@add_con(c,
        -x[d.y1tp1] + one5H * x[d.y1t] - fiveH * x[d.y2t] + fiveH * x[d.xt]
        - fiveH * x[d.y2t]^2 * x[d.y1t]
        for d in cdata)
    ExaModels.@add_con(c,
        -x[d.y2tp1] + x[d.y2t] + fiveH * x[d.y1t]
        for d in cdata)

    return ExaModels.ExaModel(c; kwargs...)
end
