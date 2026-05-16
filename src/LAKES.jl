# LAKES: Water resource management problem (90 variables, 78 constraints)
# Source: S.J. Sadjadi, University of Waterloo. SIF: Nick Gould & Sadjadi, Nov 1995.
# classification QOR2-RN-90-78
# N=6, 15 variables per time step: T(i,1..5), O(i,1..5), V(i,2), W(i,2), V(i,3), W(i,3), V(i,4)
# Variable declaration order per block i:
#   DO J=1..5: T(i,J), O(i,J) → interleaved
#   then V(i,2), W(i,2), V(i,3), W(i,3), V(i,4)
# Variable index (1-based):
#   T(i,j) = (i-1)*15 + 2*(j-1) + 1
#   O(i,j) = (i-1)*15 + 2*(j-1) + 2
#   V(i,2)=(i-1)*15+11, W(i,2)=(i-1)*15+12, V(i,3)=(i-1)*15+13
#   W(i,3)=(i-1)*15+14, V(i,4)=(i-1)*15+15
#
# Constraint ordering (from SIF GROUPS section):
#   G(i,j) for i=1..N-1, j=1..5 (conservation)
#   G(N,j) for j=1..5 (periodic)
#   For each i=1..N: A(i,1), A(i,2), A(i,3), V(i,2), W(i,2), V(i,3), W(i,3), V(i,4)
function lakes_model(; T = Float64, backend = nothing, kwargs...)
    N = 6
    nvar = 15 * N  # 90 total variables

    # Target values for T(i,j): S{i}s{j}
    Ss = T[
        202761.072  277791.816  2636.996   59987.0235  19490.4
        202703.646  277849.512  2638.1     59998.59    19555.2
        202720.536  277955.288  2639.894   60046.959   19597.6
        202808.364  278104.336  2640.906   60074.298   19652.8
        202916.46   278224.536  2641.458   60091.122   19708.8
        202953.618  278277.424  2641.458   60082.71    19706.4
    ]

    # Target values for O(i,j): R(N+i,j)
    # From SIF CONSTANTS: R(7,1)=O1o5=232.252, R(7,2..5)=O1o2..O1o5
    # R(8,j)=O2oj, ..., R(12,j)=O6oj
    Os = T[
        232.252  174.665  180.539  211.558  232.252
         83.789  173.255  179.917  210.585  215.254
         82.916  173.721  182.676  207.838  203.855
         80.134  178.654  185.917  206.416  186.308
         65.345  188.01   192.568  204.3    201.1
         72.005  193.833  196.651  204.25   241.079
    ]

    # RHS constants for G(i,j) constraints
    Gc = T[
        -22.0   -1.0   3.0  -27.2   51.5
         44.0  162.0   8.0   12.5   53.5
        -11.0   60.0  10.0   18.0   39.0
        124.0  246.0   6.0    9.7   17.2
        127.0  175.0   3.0   10.0   30.2
         78.0  156.0   3.0   14.0   23.2
    ]

    # Variable index helpers
    idxT(i, j) = (i - 1) * 15 + 2 * (j - 1) + 1
    idxO(i, j) = (i - 1) * 15 + 2 * (j - 1) + 2
    # k=1: V(i,2), k=2: W(i,2), k=3: V(i,3), k=4: W(i,3), k=5: V(i,4)
    idxVW(i, k) = (i - 1) * 15 + 10 + k

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)

    for i in 1:N
        lvar[idxVW(i, 2)] = T(0.0001)  # W(i,2)
        lvar[idxVW(i, 4)] = T(0.0001)  # W(i,3)
        lvar[idxVW(i, 5)] = T(0.0001)  # V(i,4)
    end

    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = fill(T(1.0), nvar))

    # Objective: sum_i sum_j (T(i,j) - Ss[i,j])^2 + (O(i,j) - Os[i,j])^2
    ExaModels.@add_obj(c, (x[idxT(i, j)] - Ss[i, j])^2 for i in 1:N, j in 1:5)
    ExaModels.@add_obj(c, (x[idxO(i, j)] - Os[i, j])^2 for i in 1:N, j in 1:5)

    # Conservation constraints G(i,j) for i=1..N-1, j=1..5
    for i in 1:(N - 1)
        for j in 1:5
            if j == 1
                ExaModels.@add_con(c, x[idxT(i + 1, 1)] - x[idxT(i, 1)] + x[idxO(i, 1)] - Gc[i, 1])
            elseif j == 2
                ExaModels.@add_con(c, x[idxT(i + 1, 2)] - x[idxT(i, 2)] + x[idxO(i, 2)] - x[idxO(i, 1)] - Gc[i, 2])
            elseif j == 3
                ExaModels.@add_con(c, x[idxT(i + 1, 3)] - x[idxT(i, 3)] + x[idxO(i, 3)] - x[idxO(i, 2)] - Gc[i, 3])
            elseif j == 4
                ExaModels.@add_con(c, x[idxT(i + 1, 4)] - x[idxT(i, 4)] + x[idxO(i, 4)] - x[idxO(i, 3)] - Gc[i, 4])
            else
                ExaModels.@add_con(c, x[idxT(i + 1, 5)] - x[idxT(i, 5)] + x[idxO(i, 5)] - x[idxO(i, 4)] - Gc[i, 5])
            end
        end
    end
    # Periodic constraint G(N,j) for j=1..5
    ExaModels.@add_con(c, x[idxT(1, 1)] - x[idxT(N, 1)] + x[idxO(N, 1)] - Gc[N, 1])
    ExaModels.@add_con(c, x[idxT(1, 2)] - x[idxT(N, 2)] + x[idxO(N, 2)] - x[idxO(N, 1)] - Gc[N, 2])
    ExaModels.@add_con(c, x[idxT(1, 3)] - x[idxT(N, 3)] + x[idxO(N, 3)] - x[idxO(N, 2)] - Gc[N, 3])
    ExaModels.@add_con(c, x[idxT(1, 4)] - x[idxT(N, 4)] + x[idxO(N, 4)] - x[idxO(N, 3)] - Gc[N, 4])
    ExaModels.@add_con(c, x[idxT(1, 5)] - x[idxT(N, 5)] + x[idxO(N, 5)] - x[idxO(N, 4)] - Gc[N, 5])

    # Per-i constraints: for each i=1..N:
    # A(i,1), A(i,2), A(i,3), V(i,2), W(i,2), V(i,3), W(i,3), V(i,4)
    for i in 1:N
        # A(i,1): O(i,2) - 0.0841168 * V(i,2)^2 * W(i,2)^0.5 = 0
        ExaModels.@add_con(c, x[idxO(i, 2)] - T(0.0841168) * x[idxVW(i, 1)]^2 * x[idxVW(i, 2)]^T(0.5))
        # A(i,2): O(i,3) - 0.1280849 * V(i,3)^2 * W(i,3)^0.5 = 0
        ExaModels.@add_con(c, x[idxO(i, 3)] - T(0.1280849) * x[idxVW(i, 3)]^2 * x[idxVW(i, 4)]^T(0.5))
        # A(i,3): O(i,4) - 0.2605 * V(i,4)^2.2 = 0
        ExaModels.@add_con(c, x[idxO(i, 4)] - T(0.2605) * x[idxVW(i, 5)]^T(2.2))
        # V(i,2): T(i,2)/961.6 + T(i,3)/9.2 - V(i,2) - 543.4 = 0
        ExaModels.@add_con(c, x[idxT(i, 2)] / T(961.6) + x[idxT(i, 3)] / T(9.2) - x[idxVW(i, 1)] - T(543.4))
        # W(i,2): T(i,2)/480.8 - T(i,3)/4.6 - W(i,2) = 0
        ExaModels.@add_con(c, x[idxT(i, 2)] / T(480.8) - x[idxT(i, 3)] / T(4.6) - x[idxVW(i, 2)])
        # V(i,3): T(i,3)/4.6 - V(i,3) - 543.4 = 0
        ExaModels.@add_con(c, x[idxT(i, 3)] / T(4.6) - x[idxVW(i, 3)] - T(543.4))
        # W(i,3): T(i,3)/4.6 - T(i,4)/105.15 - W(i,3) = 0
        ExaModels.@add_con(c, x[idxT(i, 3)] / T(4.6) - x[idxT(i, 4)] / T(105.15) - x[idxVW(i, 4)])
        # V(i,4): T(i,4)/105.15 - V(i,4) - 550.11 = 0
        ExaModels.@add_con(c, x[idxT(i, 4)] / T(105.15) - x[idxVW(i, 5)] - T(550.11))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
