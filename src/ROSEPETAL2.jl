# ROSEPETAL2: Rose petal problem reformulation (N=10000)
# N+1 variables: X(1..N), S=X(N+1); all free
# Objective: sum_{i=1}^N i*x[i]  (linear)
# Constraints (2*N+1 total):
#   P(I):  2*x[i] + s <= R^2 - 1 = 3  (L type, i=1..N)
#   M(I): -2*x[i] + s <= R^2 - 1 = 3  (L type, i=1..N)
#   SUM:   sum x[i]^2 - s = 0          (E type, nonlinear)
# CUTEst ordering: P(N),M(N),P(N-1),M(N-1),...,P(1),M(1), then SUM
# Start: x[i]=R^2=4.0, s=N*R^2
function rosepetal2_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    R = T(2)
    R2 = R * R
    R2m1 = R2 - T(1)
    NR2 = T(N) * R2

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N + 1; lvar = T(-Inf), uvar = T(Inf),
                       start = [fill(R2, N); NR2])

    # Objective: sum_{i=1}^N i * x[i]
    ExaModels.@add_par(c, RI, T[T(i) for i in 1:N])
    ExaModels.@add_obj(c, RI[i] * x[i] for i in 1:N)

    # CUTEst constraint ordering: P(N),M(N),P(N-1),M(N-1),...,P(1),M(1),SUM
    pm_idx = Int[i for k in N:-1:1 for i in (k, k)]
    pm_sign = T[s for k in N:-1:1 for s in (T(2), T(-2))]

    ExaModels.@add_par(c, PM_idx, pm_idx)
    ExaModels.@add_par(c, PM_sign, pm_sign)

    ExaModels.@add_con(c,
        PM_sign[i] * x[PM_idx[i]] + x[N + 1] - R2m1
        for i in 1:2*N; lcon = T(-Inf), ucon = T(0))

    # SUM constraint: sum_{i=1}^N x[i]^2 - x[N+1] = 0
    # Base constraint: -x[N+1], then augment with x[i]^2 for each i
    g = ExaModels.@add_con(c, -x[N + 1])
    ExaModels.@add_con!(c, g[1] += x[i]^2 for i in 1:N)

    return ExaModels.ExaModel(c; kwargs...)
end
