# EIGENB2: Symmetric eigenvalue decomposition (version 2)
# A = tridiagonal(2,-1), N=50 default (QQR2)
# E(I,J) N-type L2 objectives; O(I,J) E-type constraints
# E(I,J) uses EIGENA2-style: Q(J,I)*D(J) - sum_K A(I,K)*Q(J,K)
# For tridiagonal A: A(I,K) = 2 if K==I, -1 if K==I±1, else 0
function eigenb2_model(N = 50; T = Float64, backend = nothing, kwargs...)
    D_idx(j)    = j
    Q_idx(i, j) = N + (j - 1) * N + i
    NV = N + N * N

    # A(I,K) = 2 if K==I, -1 if |K-I|==1, else 0
    A_row(i, k) = k == i ? T(2) : (abs(k - i) == 1) ? T(-1) : T(0)

    x0 = zeros(T, NV)
    for j in 1:N
        x0[D_idx(j)]    = T(1.0)
        x0[Q_idx(j, j)] = T(1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # E(I,J) objective: L2 on (Q(J,I)*D(J) - sum_K A(I,K)*Q(J,K))^2
    for i in 1:N
        for j in 1:N
            qji_idx = Int32(Q_idx(j, i))
            dj      = Int32(D_idx(j))
            nz_data = [(a=A_row(i,k), qjk=Int32(Q_idx(j,k))) for k in max(1,i-1):min(N,i+1)]
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
