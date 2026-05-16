# GOULDQP2: Optimal knot placement QP, K=10000 default
# Variables: KNOT(1..K) then SPACE(1..K-1); total 2K-1 variables
# Objective: 0.5 * sum_{I=1}^{K-2} (SPACE(I+1) - SPACE(I))^2
# Constraints: SPACE(I) - KNOT(I+1) + KNOT(I) = 0 for I=1..K-1
# Variable bounds computed from geometric series A(I)
function gouldqp2_model(K = 10000; T = Float64, backend = nothing, kwargs...)
    NV = 2 * K - 1
    KNOT_idx(i) = i          # i = 1..K
    SPACE_idx(i) = K + i     # i = 1..K-1 → K+1..2K-1

    # Compute FACTOR
    # K/1000 (integer division), I2=2*(K/1000), I+1=K/1000+1
    # K>1000 = I2/(I+1) (integer division), K<1000 = 1 - K>1000
    K_over_1000 = div(K, 1000)
    I2 = 2 * K_over_1000
    Ip1 = K_over_1000 + 1
    K_gt_1000 = div(I2, Ip1)
    K_lt_1000 = 1 - K_gt_1000
    FACT1 = T(K_lt_1000) * T(1.01)
    FACT2 = T(K_gt_1000) * T(1.0001)
    FACTOR = FACT1 + FACT2

    # A(I) = ALPHA * FACTOR^(I-1) for I=1..K+1, with ALPHA=1, starting A(1)=2
    # Actually: A1=2, ALPHA=1, BETA=FACTOR initially; DO I=2..K+1: A(I)=ALPHA+BETA, BETA=BETA*FACTOR
    # So A(1)=2 (hardcoded), A(2)=1+FACTOR, A(3)=1+FACTOR+FACTOR^2,...
    # = 1 + sum_{k=0}^{I-2} FACTOR^k for I>=2
    A = zeros(T, K + 1)
    A[1] = T(2)
    beta = FACTOR
    for i in 2:(K + 1)
        A[i] = T(1) + beta
        beta *= FACTOR
    end

    # Starting point
    x0 = zeros(T, NV)
    for i in 1:K
        x0[KNOT_idx(i)] = A[i]
    end
    for i in 1:(K - 1)
        x0[SPACE_idx(i)] = A[i + 1] - A[i]
    end

    # Bounds
    lvar = zeros(T, NV)
    uvar = fill(T(Inf), NV)
    for i in 1:K
        lvar[KNOT_idx(i)] = A[i]
        uvar[KNOT_idx(i)] = A[i + 1]
    end
    for i in 1:(K - 1)
        dif = A[i + 2] - A[i]
        lvar[SPACE_idx(i)] = T(0.4) * dif
        uvar[SPACE_idx(i)] = T(0.6) * dif
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: 0.5 * sum_{I=1}^{K-2} (SPACE(I+1) - SPACE(I))^2
    ExaModels.@add_obj(c,
        T(0.5) * (x[SPACE_idx(i + 1)] - x[SPACE_idx(i)])^2
        for i in 1:(K - 2))

    # Constraints: SPACE(I) + KNOT(I) - KNOT(I+1) = 0
    ExaModels.@add_con(c,
        x[SPACE_idx(i)] + x[KNOT_idx(i)] - x[KNOT_idx(i + 1)]
        for i in 1:(K - 1))

    return ExaModels.ExaModel(c; kwargs...)
end
