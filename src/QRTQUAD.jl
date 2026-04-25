# QRTQUAD: mixed quartic and quadratic problem
# N=5000 variables (default), M=1100
# All variables X(1..N): lvar=0, uvar=10 (XR DEFAULT overridden by XL/XU for all via loop)
# Objective:
#   Linear part: sum_{i=1}^{N} (-10*i) * x[i]
#   QUART elements i=1..M:    F = P[i] * (x[i] * x[i+1])^4,  P[i] = i/M
#   QUAD elements i=M+1..N-1: F = 4*x[i]^2 + 2*x[N]^2 + x[i]*x[N]
#
# SIF: ELEMENT TYPE QUART: F = P*XY^4 where XY = X*Y
#      ELEMENT TYPE QUAD:  F = 4*X^2 + 2*Y^2 + X*Y
function qrtquad_model(N = 5000, M = 1100; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0.0), uvar = T(10.0), start = fill(T(0.0), N))
    # Linear part: coefficient of x[i] is -10*i
    ExaModels.@add_par(c, LC, T[-T(10) * i for i in 1:N])
    ExaModels.@add_obj(c, LC[i] * x[i] for i in 1:N)
    # QUART elements: P[i] * (x[i] * x[i+1])^4,  P[i] = i/M
    ExaModels.@add_par(c, PP, T[T(i) / T(M) for i in 1:M])
    ExaModels.@add_obj(c, PP[i] * (x[i] * x[i+1])^4 for i in 1:M)
    # QUAD elements: 4*x[i]^2 + 2*x[N]^2 + x[i]*x[N]  for i=M+1..N-1
    ExaModels.@add_obj(c, T(4) * x[i]^2 + T(2) * x[N]^2 + x[i] * x[N] for i in M+1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
