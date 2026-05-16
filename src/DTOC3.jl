# DTOC3: Discrete-time optimal control problem 3 (convex QP)
# N=1500 periods, 1 control, 2 state variables
# Variables: X(T) T=1..N-1, Y(T,1) and Y(T,2) T=1..N
# Objective: sum_{T=1..N-1} (2*Y(T+1,1)^2 + Y(T+1,2)^2 + 6*X(T)^2) * (2/S)
#   where S = N (SCALE = 2/S)
# Constraints:
#   TT(T,1): Y(T+1,1) - Y(T,1) - S*Y(T,2) = 0
#   TT(T,2): Y(T+1,2) - Y(T,2) + S*Y(T,1) - S*X(T) = 0
# Fixed: Y(1,1)=15, Y(1,2)=5
function dtoc3_model(N = 1500; T = Float64, backend = nothing, kwargs...)
    S  = T(N)
    invS = T(1.0 / N)
    twoInvS = T(2.0) * invS  # SCALE = 2/S

    # Variable layout: X(T) for T=1..N-1, then Y(T,1) for T=1..N, then Y(T,2) for T=1..N
    # X(T): index T (1-based)
    # Y(T,1): index (N-1) + T
    # Y(T,2): index (N-1) + N + T = 2N-1 + T
    NV = (N - 1) + N + N  # = 3N-1

    x_idx(t)    = t                    # T=1..N-1
    y1_idx(t)   = (N - 1) + t         # T=1..N
    y2_idx(t)   = (N - 1) + N + t     # T=1..N

    # Bounds: all free, Y(1,1)=15 and Y(1,2)=5 fixed
    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    lvar[y1_idx(1)] = T(15.0); uvar[y1_idx(1)] = T(15.0)
    lvar[y2_idx(1)] = T(5.0);  uvar[y2_idx(1)] = T(5.0)

    x0 = zeros(T, NV)
    x0[y1_idx(1)] = T(15.0)
    x0[y2_idx(1)] = T(5.0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: sum_{T=1..N-1} (2Y1(T+1)^2 + Y2(T+1)^2 + 6X(T)^2) * (2/S)
    # GROUP USES: O(T) uses Y1SQ(T+1) coeff 2, Y2SQ(T+1) coeff 1, XSQ(T) coeff 6
    # GROUP SCALE = 2/S → contribution = v^2 / (2/S) ... wait
    # SIF: ZN O(T) 'SCALE' 2/S  → O(T) group has SCALE = 2/S
    # For identity group type with SCALE σ on N-type group, the contribution is v/σ
    # But with elements SQ (v^2), the group value v = sum of element values
    # v = 2*Y1(T+1)^2 + Y2(T+1)^2 + 6*X(T)^2 (no rhs)
    # For identity-type objective group with scale σ: contribution = v/σ = v * S/2
    # Wait: the SCALE on N-type objective group with identity function means obj += v/σ
    # So: obj += (2*Y1(T+1)^2 + Y2(T+1)^2 + 6*X(T)^2) / (2/S) = (2*Y1(T+1)^2 + ...) * S/2
    # = S/2 * (2*Y1(T+1)^2 + Y2(T+1)^2 + 6*X(T)^2)
    # = N * Y1(T+1)^2 + (N/2)*Y2(T+1)^2 + 3*N*X(T)^2
    odata = [(
        y1tp1 = y1_idx(t + 1),
        y2tp1 = y2_idx(t + 1),
        xt    = x_idx(t),
    ) for t in 1:(N-1)]
    ExaModels.@add_obj(c,
        S * x[d.y1tp1]^2 + (S / T(2)) * x[d.y2tp1]^2 + T(3) * S * x[d.xt]^2
        for d in odata)

    # Constraints TT(T,1): -Y(T+1,1) + Y(T,1) + S*Y(T,2) = 0
    # TT(T,2): -Y(T+1,2) + Y(T,2) - S*Y(T,1) + S*X(T) = 0
    # (SIF: Y(T+1,1) - Y(T,1) - S*Y(T,2) - S*X(T)*(in TT(T,1))... let me re-read)
    # SIF for TT(T,1): XE TT(T,1) Y(T+1,1) -1.0; XE TT(T,1) Y(T,1) 1.0; ZE TT(T,1) Y(T,2) S
    #   → -Y(T+1,1) + Y(T,1) + S*Y(T,2) = 0  (rhs=0 by default)
    # SIF for TT(T,2): XE TT(T,2) Y(T+1,2) -1.0; XE TT(T,2) Y(T,2) 1.0; ZE TT(T,2) Y(T,1) -S; ZE TT(T,2) X(T) S
    #   → -Y(T+1,2) + Y(T,2) - S*Y(T,1) + S*X(T) = 0
    cdata = [(
        y1tp1 = y1_idx(t + 1),
        y2tp1 = y2_idx(t + 1),
        y1t   = y1_idx(t),
        y2t   = y2_idx(t),
        xt    = x_idx(t),
    ) for t in 1:(N-1)]
    ExaModels.@add_con(c, -x[d.y1tp1] + x[d.y1t] + S * x[d.y2t] for d in cdata)
    ExaModels.@add_con(c, -x[d.y2tp1] + x[d.y2t] - S * x[d.y1t] + S * x[d.xt] for d in cdata)

    return ExaModels.ExaModel(c; kwargs...)
end
