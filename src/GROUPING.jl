# GROUPING: Homogeneous grouping of 20 cans into 5 groups, 100 variables, 125 constraints
# Variables: X(J,K) for J=1..20, K=1..5 at index (J-1)*5+K; bounds [0,1]
# Objective: sum_{K,I} (sum_J T(J,I)*X(J,K) - XM(I))^2
# Constraints:
#   CL1(K): sum_J X(J,K) - 4 = 0 [5 equality]
#   CL2(J): sum_K X(J,K) - 1 = 0 [20 equality]
#   CNL(J,K): X(J,K)*(X(J,K)-1) = 0 [100 equality]
function grouping_model(; T = Float64, backend = nothing, kwargs...)
    POTS = 20; LOTS = 5; CAR = 2
    NV = POTS * LOTS

    X_idx(j, k) = (j - 1) * LOTS + k

    # Can characteristics
    T1 = T[4.37, 4.56, 4.26, 4.56, 4.30, 4.46, 3.84, 4.57, 4.26, 4.37,
           3.49, 4.43, 4.48, 4.01, 4.29, 4.42, 4.23, 4.42, 4.23, 3.49]
    T2 = T[5.23, 5.74, 4.93, 5.74, 5.19, 5.46, 4.65, 5.27, 5.57, 5.12,
           5.73, 5.45, 5.42, 4.05, 4.26, 4.58, 3.94, 4.18, 4.18, 5.89]
    XM = T[17.1, 20.1]

    x0 = zeros(T, NV)
    # Feasible starting point: cans 1-4 in group 1, 5-8 in group 2, ...
    for k in 1:LOTS
        for j in (4 * (k - 1) + 1):(4 * k)
            x0[X_idx(j, k)] = T(1)
        end
    end

    lvar = zeros(T, NV)
    uvar = ones(T, NV)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective OBJ(K,I): (sum_J T(J,I)*X(J,K) - XM(I))^2
    # Embed T values in named tuples
    obj_data = [(
        k  = k,
        xj = [X_idx(j, k) for j in 1:POTS],
        tj = i == 1 ? [T1[j] for j in 1:POTS] : [T2[j] for j in 1:POTS],
        xm = XM[i],
    ) for k in 1:LOTS for i in 1:CAR]
    ExaModels.@add_obj(c,
        (sum(d.tj[j] * x[d.xj[j]] for j in 1:POTS) - d.xm)^2
        for d in obj_data)

    # CL1(K): sum_J X(J,K) - 4 = 0
    cl1_data = [(xj = [X_idx(j, k) for j in 1:POTS],) for k in 1:LOTS]
    ExaModels.@add_con(c,
        sum(x[d.xj[j]] for j in 1:POTS) - T(4)
        for d in cl1_data)

    # CL2(J): sum_K X(J,K) - 1 = 0
    cl2_data = [(xk = [X_idx(j, k) for k in 1:LOTS],) for j in 1:POTS]
    ExaModels.@add_con(c,
        sum(x[d.xk[k]] for k in 1:LOTS) - T(1)
        for d in cl2_data)

    # CNL(J,K): X(J,K)*(X(J,K)-1) = 0 = X^2 - X
    ExaModels.@add_con(c,
        x[X_idx(j, k)] * (x[X_idx(j, k)] - T(1))
        for j in 1:POTS for k in 1:LOTS)

    return ExaModels.ExaModel(c; kwargs...)
end
