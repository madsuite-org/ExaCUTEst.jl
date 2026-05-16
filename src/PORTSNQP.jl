# PORTSNQP: Portfolio-type QP with extra term
# N vars x_i >= 0
# Objective:
#   sum_{i=1}^{N} (x_i - C_i)^2 where C_i = (2i-N)/N (L2 group O(I))
#   + x_N^2 / (-0.5) = -2*x_N^2 (L2 group O with scale=-0.5)
# Constraints:
#   E: sum_i x_i = 1
#   E2: sum_i C_i*x_i = 0.5
function portsnqp_model(N = 100000; T = Float64, backend = nothing, kwargs...)
    RN = T(N)
    lvar = zeros(T, N)
    uvar = fill(T(Inf), N)
    start = fill(T(0.5), N)

    Cvals = T[(T(2) * T(i) - RN) / RN for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, C, Cvals)

    # O(I) groups: (x_i - C_i)^2 for I=1..N
    ExaModels.@add_obj(c, (x[i] - C[i])^2 for i in 1:N)

    # O group: x_N^2 / (-0.5) = -2*x_N^2
    ExaModels.@add_obj(c, x[N]^2 / T(-0.5))

    # Constraint E: sum_i x_i = 1
    ExaModels.@add_con(c, sum(x[i] for i in 1:N) - T(1.0))

    # Constraint E2: sum_i C_i*x_i = 0.5
    ExaModels.@add_con(c, sum(C[i] * x[i] for i in 1:N) - T(0.5))

    return ExaModels.ExaModel(c; kwargs...)
end
