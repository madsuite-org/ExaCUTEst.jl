# TRIDIA: Shanno's tridiagonal quadratic problem (N variables, unconstrained)
# All variables free, starting at x=1.0
# Default N = 5000 (matching SIF default)
#
# SIF parameters: ALPHA=2, BETA=1, GAMMA=1, DELTA=1
#   MBETA = -BETA = -1, 1/GAMMA = 1.0
#
# Groups (all L2 type, phi(u) = u^2, contribution = v^2/sigma):
#   G(1): v = DELTA*x[1] - 1.0 = x[1] - 1, sigma = 1/GAMMA = 1.0
#         contribution = (x[1] - 1)^2
#   G(I) for I=2..N: v = MBETA*x[I-1] + ALPHA*x[I] = -x[I-1] + 2*x[I]
#                    sigma = 1/I  =>  contribution = v^2 / (1/I) = I*(2*x[I] - x[I-1])^2
#
# Objective: (x[1]-1)^2 + sum_{i=2}^{N} i*(2*x[i] - x[i-1])^2
function tridia_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))
    ExaModels.@add_obj(c, (x[1] - T(1))^2)
    ExaModels.@add_obj(c, i * (T(2) * x[i] - x[i-1])^2 for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end
