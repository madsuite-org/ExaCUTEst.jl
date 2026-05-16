# ARGLBLE: Variable-dimension rank-one linear equations (NLS feasibility problem)
# N=200 variables (default), M=400 equations (default)
# G(I) for I=1..M: sum_{j=1}^{N} I*j * x[j] = 1   (G(I) = I * sum_j j*x[j] - 1)
# All variables free, start = 1.0, rhs = 1.0
# Source: More', Garbow, Hillstrom (1981), problem 33
function arglble_model(N = 200, M = 400; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))

    # G(I) = I * (sum_{j=1}^N j * x[j]) - 1 = 0
    # Encode with data (i = I as T)
    data = [(ri = T(i),) for i in 1:M]
    ExaModels.@add_con(c,
        d.ri * sum(T(j) * x[j] for j in 1:N) - T(1)
        for d in data)

    return ExaModels.ExaModel(c; kwargs...)
end
