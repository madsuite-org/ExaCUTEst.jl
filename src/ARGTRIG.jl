# ARGTRIG: Variable-dimension trigonometric problem (N=200 variables, N equality constraints)
# Classification: NOR2-AN-V-V (no objective, N residual constraints)
# Source: More', Garbow, Hillstrom (1981), problem 26
#
# N equality constraints G(I) = 0 for I=1..N where:
#   G(I) = I*(cos(x[I]) + sin(x[I])) + sum_{j=1}^N cos(x[j]) - (N+I)
#
# ELEMENT TYPES:
#   SINCOS(XI):  cos(XI) + sin(XI)   (element SC(I) with coefficient I in group G(I))
#   COSINE(XJ):  cos(XJ)             (elements C(J) for all J in each group G(I))
#
# All free; start: x[i] = 1/N
function argtrig_model(N = 200; T = Float64, backend = nothing, kwargs...)
    start = fill(T(1) / T(N), N)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    # No objective (NOR2 classification: feasibility problem)
    # N equality constraints: G(I) = I*(cos(x[I])+sin(x[I])) + sum_j cos(x[j]) - (N+I) = 0
    # Note: loop variable i is used directly as a numeric multiplier (not wrapped in T())
    ExaModels.@add_con(c,
        i * (cos(x[i]) + sin(x[i])) + sum(cos(x[j]) for j in 1:N) - (N + i)
        for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
