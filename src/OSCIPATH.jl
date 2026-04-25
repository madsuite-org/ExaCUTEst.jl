# OSCIPATH: Nesterov's oscillating path problem
# Source: Nick Gould, Dec 2006; SIF default N=10
#
# CHEB element: F(TAU) = 2*TAU^2 - 1  (Chebyshev T2)
# GROUP TYPE PL2: F(v, P) = P*v^2
#
# Q(1) (PL2, P=0.25, linear X1, const 1): v = X1-1, contrib = 0.25*(X1-1)^2
# Q(I) for I=2..N (PL2, P=RHO=500, element CHEB(X(I-1))*(-1), linear X(I)):
#   v = X(I) - (2*X(I-1)^2 - 1) = X(I) - 2*X(I-1)^2 + 1
#   contrib = 500*(X(I) - 2*X(I-1)^2 + 1)^2
function oscipath_model(N = 10; T = Float64, backend = nothing, kwargs...)
    rho = T(500)
    start = [T(-1); fill(T(1), N - 1)]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    ExaModels.@add_obj(c, T(0.25) * (x[1] - T(1))^2)
    ExaModels.@add_obj(c, rho * (x[i] - T(2) * x[i - 1]^2 + T(1))^2 for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end
