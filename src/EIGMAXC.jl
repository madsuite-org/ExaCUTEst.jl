# EIGMAXC: Find the largest eigenvalue of Wilkinson's matrix, M=100, N=2M+1=201
# Minimize -D subject to: Q(I)*D - sum_K A(I,K)*Q(K) = 0, sum_K Q(K)^2 = 1
# Variables: D at index 1, Q(I) at index I+1
# Bounds: -1 <= all variables <= 1; Start: D=1, Q(I)=1/sqrt(N)
function eigmaxc_model(M = 100; T = Float64, backend = nothing, kwargs...)
    N = 2 * M + 1
    D_idx = 1
    Q_idx(i) = i + 1
    NV = N + 1
    sqrtN = sqrt(T(N))

    # Full Wilkinson matrix: A(I,I) = |M+1-I|, A(I,I±1) = 1
    function A_full(i, k)
        i == k ? T(abs(M + 1 - i)) : (abs(i - k) == 1) ? T(1) : T(0)
    end

    x0 = zeros(T, NV)
    x0[D_idx] = T(1)
    for i in 1:N
        x0[Q_idx(i)] = T(1) / sqrtN
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-1), uvar = T(1), start = x0)

    # Objective: minimize -D
    ExaModels.@add_obj(c, -x[D_idx])

    # E(I) equality: Q(I)*D - sum_K A(I,K)*Q(K) = 0
    e_data = [(
        qi  = Q_idx(i),
        qk  = [Q_idx(k) for k in 1:N],
        aik = [A_full(i, k) for k in 1:N],
    ) for i in 1:N]
    ExaModels.@add_con(c,
        x[d.qi] * x[D_idx] - sum(d.aik[k] * x[d.qk[k]] for k in 1:N)
        for d in e_data)

    # O equality: sum_K Q(K)^2 - 1 = 0 (single constraint)
    o_data = [(qi = [Q_idx(k) for k in 1:N],)]
    ExaModels.@add_con(c,
        sum(x[d.qi[k]]^2 for k in 1:N) - T(1)
        for d in o_data)

    return ExaModels.ExaModel(c; kwargs...)
end
