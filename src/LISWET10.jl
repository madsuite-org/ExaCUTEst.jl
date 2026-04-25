# LISWET10: k-convex approximation, g(t) = cos(pi*t), N=2000, K=2
# minimize 1/2 * sum_{i=1}^{N+K} (x_i - c_i)^2
# subject to x_{j} - 2*x_{j+1} + x_{j+2} >= 0  for j=1..N
# t_i = (i-1)/(N+K-1),  c_i = cos(pi*t_i) + 0.1*sin(i)
# All variables free (FR DEFAULT)
function liswet10_model(N = 2000, K = 2; T = Float64, backend = nothing, kwargs...)
    NK = N + K
    NK1 = NK - 1
    pi_val = T(4) * atan(T(1))
    ci = T[cos(pi_val * (T(i - 1) / T(NK1))) + T(0.1) * sin(T(i)) for i in 1:NK]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NK; lvar = T(-Inf), uvar = T(Inf), start = T(0))
    ExaModels.@add_par(c, C, ci)
    ExaModels.@add_obj(c, T(0.5) * (x[i] - C[i])^2 for i in 1:NK)
    ExaModels.@add_con(c, x[j] - T(2) * x[j+1] + x[j+2] for j in 1:N; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
