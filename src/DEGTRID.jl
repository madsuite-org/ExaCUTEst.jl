# DEGTRID: Degenerate bound-constrained convex QP with tridiagonal Hessian
# N=100000 (default), N+1 variables X(0..N), lvar=0, uvar=Inf, start=2.0
# Objective (QUADRATIC section):
#   linear: -0.5*X(0) - 1.5*X(1) - 2*X(I for I=2..N-1) - 1.5*X(N)
#   quad:   0.5*X(I)^2 for I=0..N  +  0.5*X(I)*X(I-1) for I=1..N
function degtrid_model(N = 100000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N + 1; lvar = T(0), uvar = T(Inf), start = T(2))
    # Linear terms (1-indexed: x[1]=X(0), x[2]=X(1), ..., x[N+1]=X(N))
    ExaModels.@add_obj(c, -T(0.5) * x[1])
    ExaModels.@add_obj(c, -T(1.5) * x[2])
    ExaModels.@add_obj(c, -T(2) * x[i] for i in 3:N)
    ExaModels.@add_obj(c, -T(1.5) * x[N+1])
    # Quadratic diagonal: 0.5*X(I)^2
    ExaModels.@add_obj(c, T(0.5) * x[i]^2 for i in 1:N+1)
    # Quadratic off-diagonal: SIF 'X X(I) X(I-1) 0.5' → full coeff 0.5 (not halved for off-diag)
    ExaModels.@add_obj(c, T(0.5) * x[i+1] * x[i] for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
