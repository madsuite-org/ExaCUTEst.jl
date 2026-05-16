# QR3DLS: Least-squares variant of QR3D
# Same variable/structure as QR3D but groups are N (objective) with L2 group type
# O(I,J) and F(I,J) are both objective groups with L2 type → (v - rhs)^2
function qr3dls_model(M = 20; T = Float64, backend = nothing, kwargs...)
    nQ = M * M
    nR = M * (M + 1) ÷ 2
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
    R_start = Vector{Int}(undef, M + 1)
    R_start[1] = nQ + 1
    for i in 1:M
        R_start[i + 1] = R_start[i] + (M - i + 1)
    end
    idx_R(i, j) = R_start[i] + (j - i)

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    for i in 1:M
        lvar[idx_R(i, i)] = T(0.0)
    end

    start = zeros(T, nvar)
    for i in 1:M
        start[idx_Q(i, i)] = T(1.0)
    end
    for i in 1:M-1
        start[idx_R(i, i)] = A[i, i]
        start[idx_R(i, i+1)] = A[i, i+1]
    end
    start[idx_R(M, M)] = A[M, M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Orthogonality terms: (sum_K Q(I,K)*Q(J,K) - delta(I,J))^2  for I<=J
    for i in 1:M
        for j in i:M
            rhs = i == j ? T(1.0) : T(0.0)
            ExaModels.@add_obj(c, (sum(x[idx_Q(i, k)] * x[idx_Q(j, k)] for k in 1:M) - rhs)^2)
        end
    end

    # Factorization terms: (sum_{K=1}^{J} Q(I,K)*R(K,J) - A(I,J))^2
    for i in 1:M
        for j in 1:M
            aij = A[i, j]
            ExaModels.@add_obj(c, (sum(x[idx_Q(i, k)] * x[idx_R(k, j)] for k in 1:j) - aij)^2)
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end
