# GOULDQP3: Optimal knot placement variant QP, K=10000 default
# Variables: KNOT(1..K) then SPACE(1..K-1); total 2K-1 variables
# Objective: 0.5*sum_{I=1}^{K-2} (SPACE(I+1)-SPACE(I))^2
#          + 0.5*sum_{I=1}^{K-1} (KNOT(K-I)+SPACE(I) - A(K+1-I))^2
# Constraints: SPACE(I) + KNOT(I) - KNOT(I+1) = 0 for I=1..K-1
function gouldqp3_model(K = 10000; T = Float64, backend = nothing, kwargs...)
    NV = 2 * K - 1
    KNOT_idx(i) = i
    SPACE_idx(i) = K + i

    # Compute FACTOR (same as GOULDQP2)
    K_over_1000 = div(K, 1000)
    I2 = 2 * K_over_1000
    Ip1 = K_over_1000 + 1
    K_gt_1000 = div(I2, Ip1)
    K_lt_1000 = 1 - K_gt_1000
    FACTOR = T(K_lt_1000) * T(1.01) + T(K_gt_1000) * T(1.0001)

    # A(I) values
    A = zeros(T, K + 1)
    A[1] = T(2)
    beta = FACTOR
    for i in 2:(K + 1)
        A[i] = T(1) + beta
        beta *= FACTOR
    end

    x0 = zeros(T, NV)
    for i in 1:K
        x0[KNOT_idx(i)] = A[i]
    end
    for i in 1:(K - 1)
        x0[SPACE_idx(i)] = A[i + 1] - A[i]
    end

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

    # OBJ1: 0.5*(SPACE(I+1)-SPACE(I))^2 for I=1..K-2
    ExaModels.@add_obj(c,
        T(0.5) * (x[SPACE_idx(i + 1)] - x[SPACE_idx(i)])^2
        for i in 1:(K - 2))

    # OBJ2(I): 0.5*(KNOT(K-I)+SPACE(I) - A(K+1-I))^2 for I=1..K-1
    # Constant OBJ2(I) = A(K+1-I) = A(J) where J = K+1-I
    obj2_data = [(
        knot_idx  = KNOT_idx(K - i),
        space_idx = SPACE_idx(i),
        a_val     = A[K + 1 - i],
    ) for i in 1:(K - 1)]
    ExaModels.@add_obj(c,
        T(0.5) * (x[d.knot_idx] + x[d.space_idx] - d.a_val)^2
        for d in obj2_data)

    # Constraints: SPACE(I) + KNOT(I) - KNOT(I+1) = 0
    ExaModels.@add_con(c,
        x[SPACE_idx(i)] + x[KNOT_idx(i)] - x[KNOT_idx(i + 1)]
        for i in 1:(K - 1))

    return ExaModels.ExaModel(c; kwargs...)
end
