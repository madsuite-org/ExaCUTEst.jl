# QR3D: Find QR factorization of tridiagonal matrix A
# Variables: Q(I,J) for I,J=1..M (M^2 vars), R(I,J) for 1<=I<=J<=M (M(M+1)/2 vars)
# Total: M^2 + M(M+1)/2 = M(3M+1)/2
# Constraints: O(I,J) = sum_K Q(I,K)*Q(J,K) - delta(I,J) = 0 for I<=J (M(M+1)/2 eqs)
#              F(I,J) = sum_{K<=J} Q(I,K)*R(K,J) - A(I,J) = 0 (M^2 eqs)
# A: tridiagonal: a(i,i)=2i/M, a(i,i±1)=(1-i)/M
# Variable layout: Q row-major then R upper triangular column by column
# idx_Q(I,J) = (I-1)*M + J
# idx_R(I,J) = M*M + (I-1 for cumulative uppertri) ...
# Upper triangular R: ordering by (I,J) with J>=I, row by row
# For I=1..M, J=I..M: idx = M^2 + sum_{i=1}^{I-1}(M-i+1) + (J-I+1)
function qr3d_model(M = 20; T = Float64, backend = nothing, kwargs...)
    nQ = M * M
    nR = M * (M + 1) ÷ 2
    nvar = nQ + nR

    # Compute A matrix: tridiagonal
    A = zeros(T, M, M)
    for i in 1:M
        A[i, i] = T(2 * i) / T(M)
    end
    for i in 2:M
        A[i, i-1] = T(1 - i) / T(M)
        A[i-1, i] = T(1 - i) / T(M)  # Note: same off-diagonal value
    end
    # Actually check: a(i,i+1) = (1-i)/M  (same formula for both neighbors)
    # a(1,2) = (1-1)/M = 0 -- that's what the SIF says

    idx_Q(i, j) = (i - 1) * M + j
    # Upper triangular R: for row i, columns i..M
    # offset for row i: M^2 + sum_{k=1}^{i-1}(M-k+1) = M^2 + i*(M+1) - i*(i+1)/2 - (M+1) + 1 ???
    # Simpler: precompute
    R_start = Vector{Int}(undef, M + 1)
    R_start[1] = nQ + 1
    for i in 1:M
        R_start[i + 1] = R_start[i] + (M - i + 1)
    end
    idx_R(i, j) = R_start[i] + (j - i)  # j >= i

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    for i in 1:M
        lvar[idx_R(i, i)] = T(0.0)  # R diagonal >= 0
    end

    start = zeros(T, nvar)
    # Q = identity
    for i in 1:M
        start[idx_Q(i, i)] = T(1.0)
    end
    # R = upper triangular part of A
    for i in 1:M-1
        start[idx_R(i, i)] = A[i, i]
        start[idx_R(i, i+1)] = A[i, i+1]
    end
    start[idx_R(M, M)] = A[M, M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Orthogonality constraints: sum_K Q(I,K)*Q(J,K) = delta(I,J), for I<=J
    for i in 1:M
        for j in i:M
            rhs = i == j ? T(1.0) : T(0.0)
            ExaModels.@add_con(c, sum(x[idx_Q(i, k)] * x[idx_Q(j, k)] for k in 1:M) - rhs)
        end
    end

    # Factorization constraints: sum_{K=1}^{J} Q(I,K)*R(K,J) = A(I,J), for I,J=1..M
    for i in 1:M
        for j in 1:M
            aij = A[i, j]
            # Only K=1..j contribute (R is upper triangular, R(K,J)=0 for K>J)
            ExaModels.@add_con(c, sum(x[idx_Q(i, k)] * x[idx_R(k, j)] for k in 1:j) - aij)
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end
