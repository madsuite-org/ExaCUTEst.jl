# EIGENC2: Symmetric eigenvalue decomposition (constrained optimization version 2)
# Wilkinson's matrix A: N=2M+1 with M=25, N=51 default (QQR2)
# A(I,I) = |M+1-I|, A(I,I±1) = 1
# E(I,J) N-type L2 objectives for all I,J=1..N; O(I,J) E-type constraints for I<=J
# E(I,J) uses EIGENA2-style: Q(J,I)*D(J) - sum_K A(I,K)*Q(J,K)
function eigenc2_model(M = 25; T = Float64, backend = nothing, kwargs...)
    N = 2 * M + 1
    D_idx(j)    = j
    Q_idx(i, j) = N + (j - 1) * N + i
    NV = N + N * N

    # Full Wilkinson matrix: symmetric with A(I,I) = |M+1-I|, A(I±1,I) = 1
    A_full(i, k) = i == k ? T(abs(M + 1 - i)) : (abs(i - k) == 1) ? T(1) : T(0)

    x0 = zeros(T, NV)
    for j in 1:N
        x0[D_idx(j)]    = T(1.0)
        x0[Q_idx(j, j)] = T(1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # E(I,J) objective: L2 type, for all I=1..N, J=1..N
    # E(I,J) = Q(J,I)*D(J) - sum_K A(I,K)*Q(J,K)
    for i in 1:N
        for j in 1:N
            qji_idx = Int32(Q_idx(j, i))
            dj      = Int32(D_idx(j))
            nz_data = [(a=A_full(i,k), qjk=Int32(Q_idx(j,k))) for k in max(1,i-1):min(N,i+1) if A_full(i,k) != T(0)]
            ExaModels.@add_obj(c, (x[qji_idx] * x[dj] - sum(d.a * x[d.qjk] for d in nz_data))^2)
        end
    end

    # O(I,J) constraints for I<=J
    for i in 1:N
        for j in i:N
            delt = T(i == j ? 1 : 0)
            trip_o = [(qi=Int32(Q_idx(k,i)), qj=Int32(Q_idx(k,j))) for k in 1:N]
            ExaModels.@add_con(c, sum(x[d.qi]*x[d.qj] for d in trip_o) - delt)
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end
