# INDEF: indefinite nonconvex problem (N=5000 variables, unconstrained)
# minimize sum_{i=1}^{N} x[i] + 0.5*sum_{i=2}^{N-1} cos(2*x[i] - x[N] - x[1])
# Source: Gould (1992)
# SIF: L2(I) groups with default identity type (phi(v)=v), so contribution = x[i];
#   the *T 'DEFAULT' L2 line is commented out in GROUP USES, so no L2 type is applied.
#   COS(I) groups with COS type (ALPHA*cos(v)) for i=2..N-1, where ALPHA=0.5
#   and v = 2*x[i] - x[N] - x[1]
# All variables FREE; start: x[i] = i/(N+1)
function indef_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    start_vec = T[i / (N + 1) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start_vec)
    ExaModels.@add_obj(c, x[i] for i in 1:N)
    ExaModels.@add_obj(c, T(0.5) * cos(T(2) * x[i] - x[N] - x[1]) for i in 2:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
