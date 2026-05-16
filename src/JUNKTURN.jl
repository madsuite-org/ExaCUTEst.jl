# JUNKTURN: Junkins-Turner spacecraft orientation optimal control
# N=1000, 10*(N+1)=10010 variables, 7*N=7000 equality constraints
# State: X(1..7, T=0..N), Control: U(1..3, T=0..N)
# Variable ordering: X(1,0)..X(1,N), X(2,0)..X(2,N), ..., X(7,0)..X(7,N),
#                    U(1,0)..U(1,N), U(2,0)..U(2,N), U(3,0)..U(3,N)
# Variable index: X(I,T) → (I-1)*(N+1) + T+1, I=1..7
#                 U(K,T) → 7*(N+1) + (K-1)*(N+1) + T+1, K=1..3
#
# Mesh: H = 100/N, H/2, H/4
# Constants: 6H/5=1.2H, SH=1.0909H, S1H=-0.08333H, S2H=0.18182H, -H/10, H/4
#
# Objective: trapezoidal rule sum of U_k^2 * H/2
#   = H/4 * [U1(0)^2 + U1(N)^2] + H/2 * sum_{T=1}^{N-1} [U1(T)^2 + U2(T)^2 + U3(T)^2]
#
# State constraints for T=1..N:
#   C(1,T): X(1,T) - X(1,T-1) - H/2*(−X(2,T)*X(5,T)−X(3,T)*X(6,T)−X(4,T)*X(7,T)) = 0
#   C(2,T): X(2,T) - X(2,T-1) + H/2*(−X(1,T)*X(5,T)−X(3,T)*X(7,T)+X(4,T)*X(6,T)) = 0
#   Wait: from SIF:
#     ZE C(1,T) P25(T) -H/2: X(2,T)*X(5,T) * (-H/2)
#     ZE C(1,T) P36(T) -H/2: X(3,T)*X(6,T) * (-H/2)
#     ZE C(1,T) P47(T) -H/2: X(4,T)*X(7,T) * (-H/2)
#   So C(1,T) group linear: X(1,T)*(-1) + X(1,T-1)*(1), elements: P25*(-H/2), P36*(-H/2), P47*(-H/2)
#   Constraint: X(1,T)*(-1) + X(1,T-1)*(1) + (-H/2)*(X(2,T)*X(5,T)) + (-H/2)*(X(3,T)*X(6,T))
#              + (-H/2)*(X(4,T)*X(7,T)) = 0
#   i.e.: X(1,T) - X(1,T-1) + (H/2)*(X(2,T)*X(5,T) + X(3,T)*X(6,T) + X(4,T)*X(7,T)) = 0
#
#   C(2,T): X(2,T) - X(2,T-1) - (H/2)*(X(1,T)*X(5,T)) - (H/2)*(X(3,T)*X(7,T)) + (H/2)*(X(4,T)*X(6,T)) = 0
#    from SIF: ZE C(2,T) P15 H/2, P37 H/2, P46 -H/2 → wait let me recheck
#    ZE C(2,T) P15(T) H/2, P37(T) H/2, P46(T) -H/2
#    P15 = X(1)*X(5), P37 = X(3)*X(7), P46 = X(4)*X(6)
#    Group linear: X(2,T)*(-1) + X(2,T-1)*(1)
#    Constraint = -X(2,T) + X(2,T-1) + (H/2)*X(1,T)*X(5,T) + (H/2)*X(3,T)*X(7,T) - (H/2)*X(4,T)*X(6,T) = 0
#    i.e.: X(2,T)-X(2,T-1) - (H/2)*(X(1,T)*X(5,T)+X(3,T)*X(7,T)-X(4,T)*X(6,T)) = 0
#
#   C(3,T): X(3,T)-X(3,T-1) - (H/2)*(X(1,T)*X(6,T) - X(2,T)*X(7,T) + X(4,T)*X(5,T)) = 0
#    ZE P16 H/2, P27 -H/2, P45 H/2
#    → -X(3,T) + X(3,T-1) + (H/2)*(X(1,T)*X(6,T)) - (H/2)*(X(2,T)*X(7,T)) + (H/2)*(X(4,T)*X(5,T)) = 0
#
#   C(4,T): X(4,T)-X(4,T-1) - (H/2)*(X(1,T)*X(7,T) + X(2,T)*X(6,T) - X(3,T)*X(5,T)) = 0
#    ZE P17 H/2, P26 H/2, P35 -H/2
#
#   C(5,T): X(5,T)-X(5,T-1) - H*U(1,T) + S1H*X(6,T)*X(7,T) = 0
#    Linear: X(5,T)*(-1)+X(5,T-1)*(1)+U(1,T)*H; ZE P67 S1H
#    → -X(5,T)+X(5,T-1)+H*U(1,T) + S1H*X(6,T)*X(7,T) = 0
#    i.e.: X(5,T)-X(5,T-1) - H*U(1,T) - S1H*X(6,T)*X(7,T) = 0
#
#   C(6,T): X(6,T)-X(6,T-1) - (6H/5)*U(2,T) - (-H/10)*X(5,T)*X(7,T) = 0
#    ZE P57 -H/10 (S is negative here → -(-H/10)*P57 = (H/10)*P57? No)
#    Actually ZE C(6,T) P57 -H/10 → P57 coefficient is -H/10
#    → -X(6,T)+X(6,T-1)+(6H/5)*U(2,T) + (-H/10)*X(5,T)*X(7,T) = 0
#    i.e.: X(6,T)-X(6,T-1) - (6H/5)*U(2,T) + (H/10)*X(5,T)*X(7,T) = 0
#
#   C(7,T): X(7,T)-X(7,T-1) - SH*U(3,T) - S2H*X(5,T)*X(6,T) = 0
#    ZE P56 S2H → P56 coefficient S2H
#    → -X(7,T)+X(7,T-1)+SH*U(3,T) + S2H*X(5,T)*X(6,T) = 0
#    i.e.: X(7,T)-X(7,T-1) - SH*U(3,T) - S2H*X(5,T)*X(6,T) = 0
function junkturn_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    H = T(100) / T(N)
    H2 = H / T(2)
    H4 = H / T(4)
    G65 = T(6) * H / T(5)  # 6H/5
    SH = T(1.0909) * H
    S1H = T(-0.08333) * H
    S2H = T(0.18182) * H
    H10 = H / T(10)

    # Variable index: X(I,T) → (I-1)*(N+1) + T+1 for I=1..7, T=0..N
    #                 U(K,T) → 7*(N+1) + (K-1)*(N+1) + T+1 for K=1..3
    NP1 = N + 1
    x_idx(I, t) = (I - 1) * NP1 + t + 1
    u_idx(K, t) = 7 * NP1 + (K - 1) * NP1 + t + 1
    nvar = 10 * NP1

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    start = ones(T, nvar)

    # Fixed initial state
    fixed_ic = [(1, T(1)), (2, T(0)), (3, T(0)), (4, T(0)), (5, T(0.01)), (6, T(0.005)), (7, T(0.001))]
    for (I, v) in fixed_ic
        idx = x_idx(I, 0)
        lvar[idx] = v; uvar[idx] = v; start[idx] = v
    end
    # Fixed final state
    fixed_fc = [(1, T(0.43047)), (2, T(0.70106)), (3, T(0.0923)), (4, T(0.56098)),
                (5, T(0)), (6, T(0)), (7, T(0))]
    for (I, v) in fixed_fc
        idx = x_idx(I, N)
        lvar[idx] = v; uvar[idx] = v; start[idx] = v
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Objective: trapezoidal rule for sum U_k^2 * H/2
    # = H/4 * (U1(0)^2 + U1(N)^2) + H/2 * sum_{T=1}^{N-1} (U1^2 + U2^2 + U3^2)
    # + H/2 * (U2(0)^2 + U2(N)^2 + U3(0)^2 + U3(N)^2) -- but SIF only mentions U1 at endpoints
    # From SIF: ZE OBJ U1S(0) H/4, ZE OBJ U1S(N) H/4, ZE OBJ U1S(T)..U3S(T) H/2 for T=1..N-1
    # This means U2, U3 at endpoints are NOT in objective at all!
    ExaModels.@add_obj(c, H4 * x[u_idx(1, 0)]^2)
    ExaModels.@add_obj(c, H4 * x[u_idx(1, N)]^2)
    ExaModels.@add_obj(c,
        H2 * (x[7 * NP1 + t]^2 + x[8 * NP1 + t]^2 + x[9 * NP1 + t]^2)
        for t in 1:(N-1))

    # State constraints for T=1..N
    ExaModels.@add_con(c,
        x[x_idx(1, t)] - x[x_idx(1, t-1)] +
        H2 * (x[x_idx(2, t)] * x[x_idx(5, t)] + x[x_idx(3, t)] * x[x_idx(6, t)] + x[x_idx(4, t)] * x[x_idx(7, t)])
        for t in 1:N)

    ExaModels.@add_con(c,
        x[x_idx(2, t)] - x[x_idx(2, t-1)] -
        H2 * (x[x_idx(1, t)] * x[x_idx(5, t)] + x[x_idx(3, t)] * x[x_idx(7, t)] - x[x_idx(4, t)] * x[x_idx(6, t)])
        for t in 1:N)

    ExaModels.@add_con(c,
        x[x_idx(3, t)] - x[x_idx(3, t-1)] -
        H2 * (x[x_idx(1, t)] * x[x_idx(6, t)] - x[x_idx(2, t)] * x[x_idx(7, t)] + x[x_idx(4, t)] * x[x_idx(5, t)])
        for t in 1:N)

    ExaModels.@add_con(c,
        x[x_idx(4, t)] - x[x_idx(4, t-1)] -
        H2 * (x[x_idx(1, t)] * x[x_idx(7, t)] + x[x_idx(2, t)] * x[x_idx(6, t)] - x[x_idx(3, t)] * x[x_idx(5, t)])
        for t in 1:N)

    ExaModels.@add_con(c,
        x[x_idx(5, t)] - x[x_idx(5, t-1)] - H * x[u_idx(1, t)] - S1H * x[x_idx(6, t)] * x[x_idx(7, t)]
        for t in 1:N)

    ExaModels.@add_con(c,
        x[x_idx(6, t)] - x[x_idx(6, t-1)] - G65 * x[u_idx(2, t)] + H10 * x[x_idx(5, t)] * x[x_idx(7, t)]
        for t in 1:N)

    ExaModels.@add_con(c,
        x[x_idx(7, t)] - x[x_idx(7, t-1)] - SH * x[u_idx(3, t)] - S2H * x[x_idx(5, t)] * x[x_idx(6, t)]
        for t in 1:N)

    return ExaModels.ExaModel(c; kwargs...)
end
