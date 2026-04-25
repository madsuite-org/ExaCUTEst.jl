# GILBERT: Constrained optimization problem (Conn-Gould-Toint)
# N=5000 variables, X(1) lvar=0 others free; start x[i]=(-1)^(i+1)*10
# Objective: sum_{I=1}^N ((N+1-I)/N * x[I] - 1)^2 / 2   (L2 groups, scale=2)
# Constraint: sum_I x[I]^2 - 1 = 0   (SPHERE equality constraint)
function gilbert_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    lvar = fill(T(-Inf), N)
    lvar[1] = T(0)
    start = T[T(10) * (isodd(i) ? T(1) : T(-1)) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = T(Inf), start = start)
    ExaModels.@add_par(c, AI, T[T(N + 1 - i) / T(N) for i in 1:N])
    ExaModels.@add_obj(c, (AI[i] * x[i] - T(1))^2 / T(2) for i in 1:N)
    ExaModels.@add_con(c, sum(x[i]^2 for i in 1:N) - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
