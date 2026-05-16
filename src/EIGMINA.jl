# EIGMINA: Find the smallest eigenvalue of diagonal matrix A(I,I)=I (N=100)
# Minimize D subject to: Q(I)*D - sum_K A(I,K)*Q(K) = 0, sum_K Q(K)^2 = 1
# Variables: D at index 1, Q(I) at index I+1
# Bounds: -1 <= all variables <= 1; Start: D=1, Q(I)=1/sqrt(N)
function eigmina_model(N = 100; T = Float64, backend = nothing, kwargs...)
    D_idx = 1
    Q_idx(i) = i + 1
    NV = N + 1
    sqrtN = sqrt(T(N))

    x0 = zeros(T, NV)
    x0[D_idx] = T(1)
    for i in 1:N
        x0[Q_idx(i)] = T(1) / sqrtN
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-1), uvar = T(1), start = x0)

    # Objective: minimize D
    ExaModels.@add_obj(c, x[D_idx])

    # E(I) equality: Q(I)*D - sum_K A(I,K)*Q(K) = 0
    # For diagonal A: A(I,K) = I if K==I, else 0
    ExaModels.@add_con(c,
        x[Q_idx(i)] * x[D_idx] - T(i) * x[Q_idx(i)]
        for i in 1:N)

    # O equality: sum_K Q(K)^2 - 1 = 0 (single constraint)
    o_data = [(qi = [Q_idx(k) for k in 1:N],)]
    ExaModels.@add_con(c,
        sum(x[d.qi[k]]^2 for k in 1:N) - T(1)
        for d in o_data)

    return ExaModels.ExaModel(c; kwargs...)
end
