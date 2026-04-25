# VARDIM: variable-dimension problem (N=10 default, unconstrained)
# minimize sum_{i=1}^{N} (x[i]-1)^2 + (sum_{i=1}^{N} i*x[i] - N*(N+1)/2)^2
#          + (sum_{i=1}^{N} i*x[i] - N*(N+1)/2)^4
# All variables free; start: x[i] = 1 - i/N
# Source: More', Garbow, Hillstrom (1981), problem 25
#
# NOTE: Default N=10 ("original value" in SIF). The SIF default is N=200, but the
# L4 group requires a 4D polynomial expansion (N^4 terms), which is impractical for N>50.
# Use N=10 for tests (pass "-param N=10" to CUTEst as well).
#
# SIF structure:
#   G(i) for i=1..N: L2 type, v = x[i] - 1.0 → (x[i]-1)^2
#   G(N+1): L2 type, v = sum_j(j*x[j]) - SUMJ
#     → (sum j*x_j - sumj)^2 = sum_{j,k} j*k*x_j*x_k - 2*sumj*sum_j j*x_j + sumj^2
#   G(N+2): L4 type, same v → v^4 = (A - sumj)^4 where A = sum_j j*x_j
#     → A^4 - 4*sumj*A^3 + 6*sumj^2*A^2 - 4*sumj^3*A + sumj^4
#       where A^k = sum_{i1,...,ik} i1*...*ik * x[i1]*...*x[ik] (kD loop)
#   SUMJ = N*(N+1)/2
function vardim_model(N = 10; T = Float64, backend = nothing, kwargs...)
    sumj = T(N * (N + 1) / 2)
    start = T[1 - i / N for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    # G(i) for i=1..N: (x[i] - 1)^2
    ExaModels.@add_obj(c, (x[i] - T(1))^2 for i in 1:N)
    # G(N+1): L2, (sum_j j*x_j - sumj)^2
    # = sum_{j,k} j*k*x_j*x_k - 2*sumj*(sum_j j*x_j) + sumj^2
    ExaModels.@add_obj(c, j * k * x[j] * x[k] for j in 1:N, k in 1:N)
    ExaModels.@add_obj(c, T(-2) * sumj * j * x[j] for j in 1:N)
    ExaModels.@add_obj(c, sumj^2 for _ in 1:1)
    # G(N+2): L4, (sum_j j*x_j - sumj)^4
    # Binomial expansion: A^4 - 4c*A^3 + 6c^2*A^2 - 4c^3*A + c^4
    # where A = sum_j j*x_j, c = sumj
    ExaModels.@add_obj(c, i*j*k*l * x[i]*x[j]*x[k]*x[l] for i in 1:N, j in 1:N, k in 1:N, l in 1:N)
    ExaModels.@add_obj(c, T(-4)*sumj * i*j*k * x[i]*x[j]*x[k] for i in 1:N, j in 1:N, k in 1:N)
    ExaModels.@add_obj(c, T(6)*sumj^2 * i*j * x[i]*x[j] for i in 1:N, j in 1:N)
    ExaModels.@add_obj(c, T(-4)*sumj^3 * j * x[j] for j in 1:N)
    ExaModels.@add_obj(c, sumj^4 for _ in 1:1)
    return ExaModels.ExaModel(c; kwargs...)
end
