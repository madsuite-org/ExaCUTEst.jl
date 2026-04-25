# ARGLINA: Variable dimension full rank linear problem (N=200 variables, M=400 groups)
# minimize sum_{i=1}^{N} (a*S + x[i] - 1)^2 + (M-N)*(a*S - 1)^2
# where S = sum_{j=1}^{N} x[j],  a = -2/M
# (L2 group type, no scale → contribution = v^2)
# BOUNDS: FR DEFAULT → all variables free; START: DEFAULT = 1.0
# Source: More', Garbow, Hillstrom (1981), problem 32
function arglina_model(N = 200, M = 400; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf),
                                 uvar = T(Inf),
                                 start = T(1.0))

    # a = -2/M
    a = T(-2) / T(M)

    # Groups i=1..N: v_i = a*sum_j x[j] + x[i] - 1,  contribution = v_i^2
    ExaModels.@add_obj(c, (a * sum(x[j] for j in 1:N) + x[i] - T(1))^2 for i in 1:N)

    # Groups i=N+1..M: identical, v = a*sum_j x[j] - 1,  contribution = v^2 each
    # Sum of (M-N) identical terms:
    ExaModels.@add_obj(c, T(M - N) * (a * sum(x[j] for j in 1:N) - T(1))^2)

    return ExaModels.ExaModel(c; kwargs...)
end
