# VARDIM: variable-dimension problem (N=200 variables, unconstrained)
# minimize sum_{i=1}^{N} (x[i]-1)^2 + (sum_{i=1}^{N} i*x[i] - N*(N+1)/2)^2
#          + (sum_{i=1}^{N} i*x[i] - N*(N+1)/2)^4
# All variables free; start: x[i] = 1 - i/N
# Source: More', Garbow, Hillstrom (1981), problem 25
#
# SIF structure (N=200 default):
#   G(i) for i=1..N: L2 type, v = x[i] - 1.0 → (x[i]-1)^2
#   G(N+1): L2 type, v = sum_i(i*x[i]) - SUMJ → v^2
#   G(N+2): L4 type, same v → v^4
#   SUMJ = N*(N+1)/2 (SIF: T=N*(N+1), SUMJ=0.5*T)
function vardim_model(N = 200; T = Float64, backend = nothing, kwargs...)
    sumj = T(N * (N + 1) / 2)
    start = T[1 - i / N for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    # G(i) for i=1..N: (x[i] - 1)^2
    ExaModels.@add_obj(c, (x[i] - T(1))^2 for i in 1:N)
    # G(N+1): L2 group, v = sum_j(j*x[j]) - sumj
    ExaModels.@add_obj(c, (sum(T(j) * x[j] for j in 1:N) - sumj)^2)
    # G(N+2): L4 group, same v
    ExaModels.@add_obj(c, (sum(T(j) * x[j] for j in 1:N) - sumj)^4)
    return ExaModels.ExaModel(c; kwargs...)
end
