# DEGTRIDL: Degenerate bound-constrained convex QP with tridiagonal Hessian and linear constraint
# N=100000 (default), N+1 variables X(0..N), lvar=0 (SIF default: x>=0), start=2.0
# One equality constraint: sum_{I=0}^N X(I) = N
#
# Objective (same QUADRATIC structure as DEGTRID, but different linear terms from GROUPS):
#   linear: +0.5*X(0) - 0.5*X(1) - X(I) for I=2..N-1 - 0.5*X(N)
#   quad:   0.5*X(I)^2 for I=0..N  +  0.25*X(I)*X(I-1) for I=1..N
#
# SIF: BOUNDS section has only commented-out entries → SIF default (x >= 0)
function degtridl_model(N = 100000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N + 1; lvar = T(0), uvar = T(Inf), start = T(2))
    # Linear terms from GROUPS (1-indexed: x[1]=X(0), x[2]=X(1), ..., x[N+1]=X(N))
    ExaModels.@add_obj(c, T(0.5) * x[1])
    ExaModels.@add_obj(c, -T(0.5) * x[2])
    ExaModels.@add_obj(c, -T(1) * x[i] for i in 3:N)
    ExaModels.@add_obj(c, -T(0.5) * x[N+1])
    # Quadratic diagonal: 0.5*X(I)^2
    ExaModels.@add_obj(c, T(0.5) * x[i]^2 for i in 1:N+1)
    # Quadratic off-diagonal: coefficient 0.5 → 0.5*X(I)*X(I-1) (off-diag SIF QUADRATIC convention)
    ExaModels.@add_obj(c, T(0.5) * x[i+1] * x[i] for i in 1:N)
    # Equality constraint: sum_{I=0}^N X(I) - N = 0 (residual form for CUTEst convention)
    ExaModels.@add_con(c, sum(x[i] for i in 1:N+1) - T(N))
    return ExaModels.ExaModel(c; kwargs...)
end
