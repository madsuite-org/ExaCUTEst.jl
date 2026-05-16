# HADAMALS: Finding Hadamard matrices of order N=20 (least squares formulation)
# N=20, N^2=400 variables Q(I,J) in [-1,1]
# Q(I,1) fixed: +1 for I=1..N/2, -1 for I=N/2+1..N
# Objective: sum L2 groups
#   O(I,J) for I=1..N, J=I..N: (sum_K Q(K,I)*Q(K,J) - N*delta(I,J))^2
#   S(I,J) for I=2..N, J=1..N: (Q(I,J)^2 - 1)^2
function hadamals_model(N = 20; T = Float64, backend = nothing, kwargs...)
    RN = T(N)
    N2 = N ÷ 2
    nvar = N * N

    lvar = fill(T(-1), nvar)
    uvar = fill(T(1), nvar)
    start = zeros(T, nvar)

    # Fix Q(I,1) = +1 for I=1..N/2, -1 for I=N/2+1..N
    for i in 1:N2
        lvar[(i - 1) * N + 1] = T(1)
        uvar[(i - 1) * N + 1] = T(1)
        start[(i - 1) * N + 1] = T(1)
    end
    for i in (N2+1):N
        lvar[(i - 1) * N + 1] = T(-1)
        uvar[(i - 1) * N + 1] = T(-1)
        start[(i - 1) * N + 1] = T(-1)
    end
    # Starting point: Q(I,J) = 0.9 for I<=N/2, -0.9 for I>N/2
    for j in 1:N
        for i in 1:N2
            start[(i - 1) * N + j] = T(0.9)
        end
        for i in (N2+1):N
            start[(i - 1) * N + j] = T(-0.9)
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # O(I,J) groups for I=1..N, J=I..N: (sum_K Q(K,I)*Q(K,J) - N*delta(I,J))^2
    # Q(K,I) = x[(K-1)*N + I]
    # Diagonal I==J: (sum_K Q(K,I)^2 - N)^2
    for i in 1:N
        ExaModels.@add_obj(c,
            (sum(x[(k - 1) * N + i]^2 for k in 1:N) - RN)^2)
    end

    # Off-diagonal I<J: (sum_K Q(K,I)*Q(K,J))^2
    for i in 1:N
        for j in (i+1):N
            ExaModels.@add_obj(c,
                (sum(x[(k - 1) * N + i] * x[(k - 1) * N + j] for k in 1:N))^2)
        end
    end

    # S(I,J) groups for I=2..N, J=1..N: (Q(I,J)^2 - 1)^2
    # LARGEL2 with FACTOR=1.0 is same as L2
    for i in 2:N
        ExaModels.@add_obj(c,
            (x[(i - 1) * N + j]^2 - T(1))^2
            for j in 1:N)
    end

    return ExaModels.ExaModel(c; kwargs...)
end
