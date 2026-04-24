# DEGDIAG: degenerate bound-constrained convex QP, tridiagonal Hessian (N+1 variables)
# minimize 0.5 * sum_{i=0}^{N} x_i^2
# subject to x_i >= i/(N+1)  for i = 0,...,N
# Default N = 100000 (matching SIF default)
# Source: Gould (2011)
# SIF: QUADRATIC section X X(I) X(I) 1.0 → Hessian entry H_{ii}=1, so obj += 0.5*x_i^2
function degdiag_model(N = 100000; T = Float64, backend = nothing, kwargs...)
    lvar = T[i / (N + 1) for i in 0:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N + 1; lvar = lvar, start = fill(T(2.0), N + 1))
    ExaModels.@add_obj(c, T(0.5) * x[i]^2 for i in 1:N+1)
    return ExaModels.ExaModel(c; kwargs...)
end
