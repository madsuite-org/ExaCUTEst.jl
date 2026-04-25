# VARDIM: variable-dimension problem (N=200 variables, unconstrained)
# minimize sum_{i=1}^{N} (x[i]-1)^2 + (sum_{i=1}^{N} i*x[i] - N*(N+1)/2)^2
#          + (sum_{i=1}^{N} i*x[i] - N*(N+1)/2)^4
# All variables free; start: x[i] = 1 - i/N
# Source: More', Garbow, Hillstrom (1981), problem 25
#
# SIF structure:
#   G(i) for i=1..N: L2 type, v = x[i] - 1.0 → (x[i]-1)^2
#   G(N+1): L2 type, v = sum_i(i*x[i]) - N*(N+1)/2 → v^2
#   G(N+2): L4 type, same v → v^4
#   SUMJ = N*(N+1)/2 (computed as 0.5*N*(N+1) in SIF via RI/R*/RM)
function vardim_model(N = 200; T = Float64, backend = nothing, kwargs...)
    sumj = T(N * (N + 1) / 2)
    start = T[1 - i / N for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    # G(i) for i=1..N: (x[i] - 1)^2
    ExaModels.@add_obj(c, (x[i] - T(1))^2 for i in 1:N)
    # G(N+1): L2, v = sum_i(i*x[i]) - sumj
    # G(N+2): L4, same v
    # Represent using @add_par for the weighted sum
    w = T[i for i in 1:N]
    ExaModels.@add_par(c, iw, w)
    ExaModels.@add_obj(c, (sum(iw[i] * x[i] for i in 1:N) - sumj)^2)
    ExaModels.@add_obj(c, (sum(iw[i] * x[i] for i in 1:N) - sumj)^4)
    return ExaModels.ExaModel(c; kwargs...)
end
