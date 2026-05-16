# QR3DBD: QR factorization with banded R structure imposed
# Q is M×M (M^2 vars), R is banded upper triangular with bandwidth 2
# R variables: R(I,I), R(I,I+1), R(I,I+2) for I=1..M-2; R(M-1,M-1), R(M-1,M), R(M,M)
# Total R vars: 3*(M-2) + 3 = 3M-3
# n = M^2 + 3*(M-1) = M^2 + 3M - 3
# Constraints: O(I,J) for I<=J (M(M+1)/2) + F(I,J) for all I,J (M^2)
# For F(I,J): sum_{K=max(1,J-2)}^{J} Q(I,K)*R(K,J)
function qr3dbd_model(M = 20; T = Float64, backend = nothing, kwargs...)
    nQ = M * M
    # R variables layout: row by row, only bands
    # I=1: R(1,1), R(1,2), R(1,3)
    # I=2: R(2,2), R(2,3), R(2,4)
    # ...
    # I=M-2: R(M-2,M-2), R(M-2,M-1), R(M-2,M)
    # I=M-1: R(M-1,M-1), R(M-1,M)
    # I=M: R(M,M)
    # Total: 3*(M-2) + 2 + 1 = 3M-3

    nR = 3 * (M - 1) - (M >= 3 ? 0 : 0)  # simplified: 3*(M-2)+3 = 3M-3
    nR = 3 * M - 3
    nvar = nQ + nR

    A = zeros(T, M, M)
    for i in 1:M
        A[i, i] = T(2 * i) / T(M)
    end
    for i in 2:M
        A[i, i-1] = T(1 - i) / T(M)
        A[i-1, i] = T(1 - i) / T(M)
    end

    idx_Q(i, j) = (i - 1) * M + j

    # R index map: store R(i,j) index for valid (i,j)
    R_idx = Dict{Tuple{Int,Int},Int}()
    cnt = nQ
    for i in 1:M-2
        cnt += 1; R_idx[(i, i)] = cnt
        cnt += 1; R_idx[(i, i+1)] = cnt
        cnt += 1; R_idx[(i, i+2)] = cnt
    end
    cnt += 1; R_idx[(M-1, M-1)] = cnt
    cnt += 1; R_idx[(M-1, M)] = cnt
    cnt += 1; R_idx[(M, M)] = cnt

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    for i in 1:M
        if haskey(R_idx, (i, i))
            lvar[R_idx[(i, i)]] = T(0.0)
        end
    end

    start = zeros(T, nvar)
    for i in 1:M
        start[idx_Q(i, i)] = T(1.0)
    end
    for i in 1:M-1
        if haskey(R_idx, (i, i))
            start[R_idx[(i, i)]] = A[i, i]
        end
        if haskey(R_idx, (i, i+1))
            start[R_idx[(i, i+1)]] = A[i, i+1]
        end
    end
    if haskey(R_idx, (M, M))
        start[R_idx[(M, M)]] = A[M, M]
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Orthogonality constraints: sum_K Q(I,K)*Q(J,K) = delta(I,J)
    for i in 1:M
        for j in i:M
            rhs = i == j ? T(1.0) : T(0.0)
            ExaModels.@add_con(c, sum(x[idx_Q(i, k)] * x[idx_Q(j, k)] for k in 1:M) - rhs)
        end
    end

    # Factorization constraints: sum_{K: R(K,J) exists and K<=J} Q(I,K)*R(K,J) = A(I,J)
    for i in 1:M
        # F(I,1): only K=1
        ExaModels.@add_con(c, x[idx_Q(i, 1)] * x[R_idx[(1, 1)]] - A[i, 1])
        # F(I,2): K=1,2
        ExaModels.@add_con(c, x[idx_Q(i, 1)] * x[R_idx[(1, 2)]] + x[idx_Q(i, 2)] * x[R_idx[(2, 2)]] - A[i, 2])
        # F(I,J) for J=3..M: K=J-2,J-1,J
        for j in 3:M
            ExaModels.@add_con(c,
                x[idx_Q(i, j-2)] * x[R_idx[(j-2, j)]] +
                x[idx_Q(i, j-1)] * x[R_idx[(j-1, j)]] +
                x[idx_Q(i, j)]   * x[R_idx[(j,   j)]] - A[i, j])
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end
