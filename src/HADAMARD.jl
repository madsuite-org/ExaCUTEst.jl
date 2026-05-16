# HADAMARD: Finding Hadamard matrices of order N=20
# 401 variables: MAXABSQ=x[1], Q(I,J)=x[1+(I-1)*N+J] for I,J=1..N, all free
# Objective: minimize MAXABSQ
# O(I,J) constraints (E): sum_{K=1}^N Q(K,I)*Q(K,J) = N*(I==J)
#   Diagonal: sum_K Q(K,I)^2 = N (N constraints)
#   Off-diagonal: sum_K Q(K,I)*Q(K,J) = 0 (N*(N-1)/2 = 190 constraints)
# L(I,J) constraints (G): MAXABSQ + Q(I,J) >= 0  (N^2=400 constraints)
# U(I,J) constraints (G): MAXABSQ - Q(I,J) >= 0  (N^2=400 constraints)
# Total: N + N(N-1)/2 + 2N^2 = 20+190+800 = 1010 constraints
function hadamard_model(N = 20; T = Float64, backend = nothing, kwargs...)
    RN = T(N)
    nvar = N * N + 1

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    start = fill(T(1), nvar)
    lvar[1] = T(0)  # MAXABSQ >= 0
    start[1] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Objective: minimize MAXABSQ = x[1]
    ExaModels.@add_obj(c, x[1])

    # Diagonal O(I,J) constraints for I==J: sum_K Q(K,I)^2 - N = 0
    # Q(K,I) = x[1 + (K-1)*N + I]
    for i in 1:N
        ExaModels.@add_con(c,
            sum(x[1 + (k - 1) * N + i]^2 for k in 1:N) - RN)
    end

    # Off-diagonal O(I,J) constraints for I<J: sum_K Q(K,I)*Q(K,J) = 0
    for i in 1:N
        for j in (i+1):N
            ExaModels.@add_con(c,
                sum(x[1 + (k - 1) * N + i] * x[1 + (k - 1) * N + j] for k in 1:N))
        end
    end

    # L(I,J) constraints (G type: >= 0): MAXABSQ + Q(I,J) >= 0
    ExaModels.@add_con(c,
        x[1] + x[1 + (i - 1) * N + j]
        for i in 1:N for j in 1:N;
        lcon = T(0), ucon = T(Inf))

    # U(I,J) constraints (G type: >= 0): MAXABSQ - Q(I,J) >= 0
    ExaModels.@add_con(c,
        x[1] - x[1 + (i - 1) * N + j]
        for i in 1:N for j in 1:N;
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
