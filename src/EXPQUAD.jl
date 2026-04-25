# EXPQUAD: mixed exponential and quadratic problem
# N=1200 variables (default), M=100
# Variables X(1..M): lvar=0, uvar=10; X(M+1..N): free (XR DEFAULT)
# Objective:
#   Linear part: sum_{i=1}^{N} (-10*i) * x[i]
#   EXP elements i=1..M:   F = exp(0.1 * P[i] * x[i] * x[i+1]),  P[i] = i/M
#   QUAD elements i=M+1..N-1: F = 4*x[i]^2 + 2*x[N]^2 + x[i]*x[N]
#
# SIF: ELEMENT TYPE EXP: F = exp(0.1*P*X*Y)
#      ELEMENT TYPE QUAD: F = 4*X^2 + 2*Y^2 + X*Y
function expquad_model(N = 1200, M = 100; T = Float64, backend = nothing, kwargs...)
    lvar = T[i <= M ? T(0.0) : T(-Inf) for i in 1:N]
    uvar = T[i <= M ? T(10.0) : T(Inf) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = fill(T(0.0), N))
    # Linear part: coefficient of x[i] is -10*i
    ExaModels.@add_par(c, LC, T[-T(10) * i for i in 1:N])
    ExaModels.@add_obj(c, LC[i] * x[i] for i in 1:N)
    # EXP elements: exp(0.1 * P[i] * x[i] * x[i+1]),  P[i] = i/M
    ExaModels.@add_par(c, PP, T[T(i) / T(M) for i in 1:M])
    ExaModels.@add_obj(c, exp(T(0.1) * PP[i] * x[i] * x[i+1]) for i in 1:M)
    # QUAD elements: 4*x[i]^2 + 2*x[N]^2 + x[i]*x[N]  for i=M+1..N-1
    ExaModels.@add_obj(c, T(4) * x[i]^2 + T(2) * x[N]^2 + x[i] * x[N] for i in M+1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
