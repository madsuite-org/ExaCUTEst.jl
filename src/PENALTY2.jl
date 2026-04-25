# PENALTY2: Penalty function II
# Source: Problem 24 in More', Garbow, Hillstrom (1981)
# SIF: N=200, M=2N=400, A=1e-5, B=1, all variables free, start x[i]=0.5
#
# G(1): scale=1, L2, X(1) coeff 1, constant 0.2
#   contribution = (x[1] - 0.2)^2
# G(I) for I=2..N: scale=1/A=100000, L2, elements E10(x[I])+E10(x[I-1]), constant=exp(I/10)+exp((I-1)/10)
#   contribution = (exp(0.1*x[I]) + exp(0.1*x[I-1]) - YI)^2 / 100000
# G(I) for I=N+1..M-1: scale=1/A=100000, L2, element E10(x[I-N+1]), constant=exp(-0.1)
#   (I-N+1 ranges from 2 to N, so j=2..N)
#   contribution = (exp(0.1*x[j]) - exp(-0.1))^2 / 100000
# G(M): scale=1, L2, elements SQ(x[J]) with coeff w_J=(N-J+1), constant=1
#   contribution = (sum_{J=1}^N (N-J+1)*x[J]^2 - 1)^2
#
# G(M) expansion: (A-1)^2 = sum_{j,k} w_j*w_k*x_j^2*x_k^2 - 2*sum_j w_j*x_j^2 + 1
# Uses Iterators.product(1:N, 1:N) for double sum to avoid deep type nesting.
function penalty2_model(N = 200; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(T(-Inf), N), uvar = fill(T(Inf), N),
                       start = fill(T(0.5), N))

    # Group G(1): (x[1] - 0.2)^2
    ExaModels.@add_obj(c, (x[1] - T(0.2))^2)

    # Groups G(2..N): (exp(0.1*x[I]) + exp(0.1*x[I-1]) - YI)^2 / 100000
    # yi[k] = exp(I/10) + exp((I-1)/10) for I=k+1 (k=1..N-1)
    ExaModels.@add_par(c, yi, T[exp(i / 10) + exp((i - 1) / 10) for i in 2:N])
    ExaModels.@add_obj(c, (exp(T(0.1) * x[i]) + exp(T(0.1) * x[i-1]) - yi[i-1])^2 / T(100000) for i in 2:N)

    # Groups G(N+1..M-1): (exp(0.1*x[j]) - exp(-0.1))^2 / 100000 for j=2..N
    ExaModels.@add_obj(c, (exp(T(0.1) * x[j]) - exp(T(-0.1)))^2 / T(100000) for j in 2:N)

    # Group G(M): (sum_{J=1}^N w_J*x[J]^2 - 1)^2  where w_J = N-J+1
    # Expand: sum_{j,k} wj*wk*xj^2*xk^2 - 2*sum_j wj*xj^2 + 1
    ExaModels.@add_par(c, wt, T[N - j + 1 for j in 1:N])
    ExaModels.@add_obj(c, wt[j] * wt[k] * x[j]^2 * x[k]^2 for (j, k) in Iterators.product(1:N, 1:N))
    ExaModels.@add_obj(c, T(-2) * wt[i] * x[i]^2 for i in 1:N)
    c, _ = ExaModels.add_obj(c, ExaModels.Constant(T(1.0)))

    return ExaModels.ExaModel(c; kwargs...)
end
