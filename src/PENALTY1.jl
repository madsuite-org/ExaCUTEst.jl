# PENALTY1: Penalty function I
# Source: Problem 23 in More', Garbow, Hillstrom (1981)
# SIF: N=1000 free variables, unconstrained, start x[i]=float(i)
#
# G(I) for I=1..N: scale σ=100000, L2 type, linear X(I) coeff 1, constant 1
#   contribution = (x[I]-1)^2 / 100000
# G(M=N+1): no scale (σ=1), L2 type, elements SQ(x[I]) for I=1..N (coeff=1), constant 0.25
#   contribution = (sum_{i=1}^N x[i]^2 - 0.25)^2
#
# Expansion: (A - c)^2 = sum_{i,j} x_i^2*x_j^2 - 2c*sum_i x_i^2 + c^2,  c=0.25
# Uses Iterators.product(1:N, 1:N) for the double sum to avoid deep type nesting.
function penalty1_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(T(-Inf), N), uvar = fill(T(Inf), N),
                       start = T[i for i in 1:N])
    # Groups G(1)..G(N): (x[i] - 1)^2 / 100000
    ExaModels.@add_obj(c, (x[i] - T(1))^2 / T(100000) for i in 1:N)
    # Group G(M): (sum_{i=1}^N x[i]^2 - 0.25)^2 expanded into separable terms
    ExaModels.@add_obj(c, x[i]^2 * x[j]^2 for (i, j) in Iterators.product(1:N, 1:N))
    ExaModels.@add_obj(c, T(-0.5) * x[i]^2 for i in 1:N)
    c, _ = ExaModels.add_obj(c, ExaModels.Constant(T(0.0625)))
    return ExaModels.ExaModel(c; kwargs...)
end
