# DEGTRID2: Degenerate bound-constrained convex QP with tridiagonal Hessian (variant 2)
# N=100000, N+1 variables; X(0) has lvar=0, X(1..N) have lvar=1.0, uvar=Inf, start=2.0
# Same quadratic objective as DEGTRID but different bounds
function degtrid2_model(N = 100000; T = Float64, backend = nothing, kwargs...)
    lvar = T[i == 1 ? T(0) : T(1) for i in 1:N+1]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N + 1; lvar = lvar, uvar = T(Inf), start = T(2))
    ExaModels.@add_obj(c, -T(0.5) * x[1])
    ExaModels.@add_obj(c, -T(1.5) * x[2])
    ExaModels.@add_obj(c, -T(2) * x[i] for i in 3:N)
    ExaModels.@add_obj(c, -T(1.5) * x[N+1])
    ExaModels.@add_obj(c, T(0.5) * x[i]^2 for i in 1:N+1)
    ExaModels.@add_obj(c, T(0.5) * x[i+1] * x[i] for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
