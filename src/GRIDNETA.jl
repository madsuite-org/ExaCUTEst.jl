# GRIDNETA-F: Nonlinear network problems on a square grid
# Source: Ph.L. Toint and D. Tuyttens,
#   "On large scale nonlinear network optimization",
#   Mathematical Programming B, vol. 48(1), pp.125-159, 1990.
# SIF input: Ph. Toint, March 1990; minor correction Ph. Shott, Jan 1995.
#
# Grid: NS×NS nodes where NS = 2L+2; arcs are horizontal (right) and vertical (up).
# N = 2*NS*(NS-1) variables (arc flows).
#
# Variable layout (1-indexed), P = 2*NS - 1:
#   Row i (i=1..NS-1): (NS-1) right arcs at (i-1)*P+2j-1 (j=1..NS-1)
#                    + NS up arcs at (i-1)*P+2j (j=1..NS-1) and (i-1)*P+2(NS-1)+1 (j=NS)
#   Row NS (top):    (NS-1) right arcs at (NS-1)*P+j (j=1..NS-1)
#
# Constraints: flow conservation C(I,J) for I,J=1..NS
#   (net outflow at each node = source/sink)
#   CONSTANTS: C(1,1)=10.0, C(NS,NS)=-10.0, all others=0.
#
# Objective: sum_k coef[k]*X(k)^2 (all variants)
#   GRIDNETD-F additionally add: (A/100)*sum_{k=1}^{N-1} sqrt(1+2*X(k)*(X(k)-X(k+1))+X(k+1)^2)
#
# Variants differ in BOUNDS:
#   A/D: MOD3 fixed to 0, odd bottom arcs >= R, top N-2L..N arcs >= R,
#        side arcs >= R, NE corner arcs in [4,5]
#   B/E: all free
#   C/F: MOD3 arcs >= R

function _gridnet_coefs(L::Int)
    NS = 2 * L + 2
    P  = 2 * NS - 1
    N  = 2 * NS * (NS - 1)
    LN10 = 2.302585093
    BETA = 2.0 * LN10 / (L - 1)

    coef = zeros(N)

    # TW = (NS-2)*P + 2  (first index in top row, minus (P-1))
    TW  = (NS - 2) * P + 2
    HTW = TW + P - 1

    # Bottom outside: J=1..P-2 step 4
    for J in 1:4:(P - 2)
        coef[J] += 0.01
    end

    # Side outside: IW=2..TW step 2P; IE = IW + P - 2
    for IW in 2:(2 * P):TW
        coef[IW]           += 0.01
        coef[IW + P - 2]   += 0.01
    end

    # Top outside: J=HTW..N step 2
    for J in HTW:2:N
        coef[J] += 0.01
    end

    # Cycles JK=1..L
    TE = HTW - 1
    for JK in 1:L
        AS     = exp((JK - 1) * BETA) * 0.01
        VW     = 4 * JK
        IW     = VW
        P2JK_1 = (2 * JK - 1) * P
        IB     = P2JK_1 + 1

        for K in 1:(NS - 1)
            coef[IW]       += AS
            coef[IW + 2]   += AS
            coef[IB]       += AS
            coef[IB + P]   += AS
            IW += P
            IB += 2
        end

        coef[VW - 1]         += AS  # bottom of vertical cycle
        coef[TE + 2 * JK]    += AS  # top of vertical cycle
        coef[P2JK_1 + 2]     += AS  # west of horizontal cycle
        coef[P2JK_1 + P]     += AS  # east of horizontal cycle
    end

    return coef
end

function _gridnet_model(L::Int, has_sqr::Bool, bounds_kind::Symbol;
                        T = Float64, backend = nothing, kwargs...)
    NS    = 2 * L + 2
    P     = 2 * NS - 1
    N     = 2 * NS * (NS - 1)
    R_val = T(0.1)

    # Arc index functions (1-based)
    R_arc(i, j) = i < NS ? (i - 1) * P + 2 * j - 1 : (NS - 1) * P + j
    U_arc(i, j) = j < NS ? (i - 1) * P + 2 * j     : (i - 1) * P + 2 * (NS - 1) + 1

    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf),  N)
    x0   = zeros(T, N)

    if bounds_kind === :A || bounds_kind === :D
        # Fix every 3rd arc (starting at 3) to 0
        for i in 3:3:N
            lvar[i] = T(0); uvar[i] = T(0)
        end
        # Bottom boundary: odd indices 1,3,...,P-1 (XP overrides MOD3 upper bound)
        for i in 1:2:(P - 1)
            lvar[i] = R_val; uvar[i] = T(Inf)
        end
        # Top boundary: N-2L..N (XP overrides MOD3 upper bound)
        for i in (N - 2 * L):N
            lvar[i] = R_val; uvar[i] = T(Inf)
        end
        # Side boundaries: column arcs KW=2, KE=P, step P each row (XP)
        KW = 2; KE = P
        for _ in 1:(NS - 1)
            lvar[KW] = R_val; uvar[KW] = T(Inf)
            lvar[KE] = R_val; uvar[KE] = T(Inf)
            KW += P; KE += P
        end
        # NE corner arcs in [4,5]
        for idx in [N - 3*L - 8, N - L - 3, N - 1, N]
            lvar[idx] = T(4); uvar[idx] = T(5)
        end

    elseif bounds_kind === :C || bounds_kind === :F
        # MOD3 arcs get lower bound R
        for i in 3:3:N
            lvar[i] = R_val
        end
    end
    # :B or :E: all free (default -Inf..Inf)

    coef = _gridnet_coefs(L)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = x0)

    # === Quadratic objective: sum_k coef[k] * x[k]^2 ===
    obj_data = [(k = k, w = T(coef[k])) for k in 1:N if coef[k] != 0.0]
    ExaModels.@add_obj(c, d.w * x[d.k]^2 for d in obj_data)

    # === SQR term (GRIDNETD-F): (A/100)*sum_k sqrt(1+2*x[k]*(x[k]-x[k+1])+x[k+1]^2) ===
    if has_sqr
        A100     = T(0.01)
        sqr_data = [(k = k, kp1 = k + 1) for k in 1:(N - 1)]
        ExaModels.@add_obj(c,
            A100 * sqrt(T(1) + T(2) * x[d.k] * (x[d.k] - x[d.kp1]) + x[d.kp1]^2)
            for d in sqr_data)
    end

    # === Constraints: flow conservation at each node C(I,J) ===
    # Encoded as: (net outflow expression) - constant = 0

    # C(1,1): R(1,1) + U(1,1) - 10 = 0
    ExaModels.@add_con(c, x[R_arc(1, 1)] + x[U_arc(1, 1)] - T(10))

    # C(1,J) J=2..NS-1: R(1,J) + U(1,J) - R(1,J-1) = 0
    bot_mid_data = [(rJ = R_arc(1, J), uJ = U_arc(1, J), rJm1 = R_arc(1, J - 1))
                    for J in 2:(NS - 1)]
    ExaModels.@add_con(c, x[d.rJ] + x[d.uJ] - x[d.rJm1] for d in bot_mid_data)

    # C(1,NS): U(1,NS) - R(1,NS-1) = 0
    ExaModels.@add_con(c, x[U_arc(1, NS)] - x[R_arc(1, NS - 1)])

    # C(I,1) I=2..NS-1: R(I,1) + U(I,1) - U(I-1,1) = 0
    mid_left_data = [(rI1 = R_arc(I, 1), uI1 = U_arc(I, 1), uIm11 = U_arc(I - 1, 1))
                     for I in 2:(NS - 1)]
    ExaModels.@add_con(c, x[d.rI1] + x[d.uI1] - x[d.uIm11] for d in mid_left_data)

    # C(I,J) I=2..NS-1, J=2..NS-1: R(I,J)+U(I,J)-R(I,J-1)-U(I-1,J)=0
    mid_int_data = [(rIJ   = R_arc(I, J),     uIJ    = U_arc(I, J),
                     rIJm1 = R_arc(I, J - 1), uIm1J  = U_arc(I - 1, J))
                    for I in 2:(NS - 1) for J in 2:(NS - 1)]
    ExaModels.@add_con(c, x[d.rIJ] + x[d.uIJ] - x[d.rIJm1] - x[d.uIm1J]
                       for d in mid_int_data)

    # C(I,NS) I=2..NS-1: U(I,NS) - R(I,NS-1) - U(I-1,NS) = 0
    mid_right_data = [(uINS   = U_arc(I, NS), rINSm1  = R_arc(I, NS - 1),
                       uIm1NS = U_arc(I - 1, NS))
                      for I in 2:(NS - 1)]
    ExaModels.@add_con(c, x[d.uINS] - x[d.rINSm1] - x[d.uIm1NS]
                       for d in mid_right_data)

    # C(NS,1): R(NS,1) - U(NS-1,1) = 0
    ExaModels.@add_con(c, x[R_arc(NS, 1)] - x[U_arc(NS - 1, 1)])

    # C(NS,J) J=2..NS-1: R(NS,J) - R(NS,J-1) - U(NS-1,J) = 0
    top_mid_data = [(rNSJ  = R_arc(NS, J), rNSJm1 = R_arc(NS, J - 1),
                     uNSm1J = U_arc(NS - 1, J))
                    for J in 2:(NS - 1)]
    ExaModels.@add_con(c, x[d.rNSJ] - x[d.rNSJm1] - x[d.uNSm1J]
                       for d in top_mid_data)

    # C(NS,NS): -R(NS,NS-1) - U(NS-1,NS) + 10 = 0
    ExaModels.@add_con(c, -x[R_arc(NS, NS - 1)] - x[U_arc(NS - 1, NS)] + T(10))

    return ExaModels.ExaModel(c; kwargs...)
end

function gridneta_model(L = 30; T = Float64, backend = nothing, kwargs...)
    _gridnet_model(L, false, :A; T = T, backend = backend, kwargs...)
end

function gridnetb_model(L = 30; T = Float64, backend = nothing, kwargs...)
    _gridnet_model(L, false, :B; T = T, backend = backend, kwargs...)
end

function gridnetc_model(L = 30; T = Float64, backend = nothing, kwargs...)
    _gridnet_model(L, false, :C; T = T, backend = backend, kwargs...)
end

function gridnetd_model(L = 30; T = Float64, backend = nothing, kwargs...)
    _gridnet_model(L, true, :D; T = T, backend = backend, kwargs...)
end

function gridnete_model(L = 30; T = Float64, backend = nothing, kwargs...)
    _gridnet_model(L, true, :E; T = T, backend = backend, kwargs...)
end

function gridnetf_model(L = 30; T = Float64, backend = nothing, kwargs...)
    _gridnet_model(L, true, :F; T = T, backend = backend, kwargs...)
end
