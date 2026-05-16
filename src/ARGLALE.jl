# ARGLALE: Variable-dimension full rank linear equations (NLS feasibility problem)
# N=200 variables (default), M=400 equations (default)
# G(I) for I=1..N: sum_{j=1}^{N} c_j * x[j] = 1
#   where c_I = 1 - 2/M,  c_{j!=I} = -2/M
#   equivalently: x[I] - (2/M)*sum_{j=1}^{N} x[j] = 1
# G(I) for I=N+1..M: -(2/M)*sum_{j=1}^{N} x[j] = 1
# All variables free, start = 1.0, rhs = 1.0
# Source: More', Garbow, Hillstrom (1981), problem 32
function arglale_model(N = 200, M = 400; T = Float64, backend = nothing, kwargs...)
    c2M = T(2) / T(M)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))

    # G(I) for I=1..N: x[I] - c2M*sum_{j=1}^N x[j] - 1 = 0
    # Use data with index i
    data_diag = [(i = i,) for i in 1:N]
    ExaModels.@add_con(c,
        x[d.i] - c2M * sum(x[j] for j in 1:N) - T(1)
        for d in data_diag)

    # G(I) for I=N+1..M: -c2M*sum_{j=1}^N x[j] - 1 = 0
    # These are M-N identical constraints; encode each separately
    data_tail = [(k = k,) for k in N+1:M]
    ExaModels.@add_con(c,
        T(-1) * c2M * sum(x[j] for j in 1:N) - T(1)
        for d in data_tail)

    return ExaModels.ExaModel(c; kwargs...)
end
