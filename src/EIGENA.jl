# EIGENA: Symmetric eigenvalue decomposition A = Q' D Q
# A = diag(1, 2, ..., N), N=50 default (classification NOR2: all constraints, no objective)
# Variables: D(J) eigenvalues (J=1..N) + Q(I,J) orthogonal matrix (I,J=1..N)
# Variable layout: D(J) at index J; Q(I,J) at index N + (J-1)*N + I
# Constraints (upper triangle I<=J, loop order: DO J=1..N, DO I=1..J):
#   E(I,J): sum_K Q(K,I)*D(K)*Q(K,J) = A(I,J)
#   O(I,J): sum_K Q(K,I)*Q(K,J) = delta(I,J)
# Matrix A is diagonal: A(I,I) = I, A(I,J) = 0 for I<J
function eigena_model(N = 50; T = Float64, backend = nothing, kwargs...)
    D_idx(j)    = j
    Q_idx(i, j) = N + (j - 1) * N + i
    NV = N + N * N

    x0 = zeros(T, NV)
    for j in 1:N
        x0[D_idx(j)]    = T(1.0)
        x0[Q_idx(j, j)] = T(1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # E constraints: sum_K Q(K,I)*D(K)*Q(K,J) - A(I,J) = 0
    # Loop order matches SIF GROUPS: DO J=1..N, DO I=1..J
    for j in 1:N
        for i in 1:j
            aij = T(i == j ? i : 0)
            trip_e = [(qi=Int32(Q_idx(k,i)), dk=Int32(D_idx(k)), qj=Int32(Q_idx(k,j))) for k in 1:N]
            ExaModels.@add_con(c, sum(x[d.qi]*x[d.dk]*x[d.qj] for d in trip_e) - aij)
        end
    end

    # O constraints: sum_K Q(K,I)*Q(K,J) - delta(I,J) = 0
    # Loop order matches SIF GROUPS: DO J=1..N, DO I=1..J
    for j in 1:N
        for i in 1:j
            delt = T(i == j ? 1 : 0)
            trip_o = [(qi=Int32(Q_idx(k,i)), qj=Int32(Q_idx(k,j))) for k in 1:N]
            ExaModels.@add_con(c, sum(x[d.qi]*x[d.qj] for d in trip_o) - delt)
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end
