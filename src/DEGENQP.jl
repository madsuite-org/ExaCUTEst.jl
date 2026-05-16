function degenqp_model(N = 10; T = Float64, backend = nothing, kwargs...)
    # Objective: sum_i [X(i) + (i/N)*X(i)^2]
    obj_data = [(i = i, p = T(i) / T(N)) for i in 1:N]

    # E(I) for I=1,3,...,N-1: X(I) - X(I+1), range [0,2]
    e_data = [(i = i, ip1 = i + 1) for i in 1:2:N-1]

    # C(I,J,K) for 1 ≤ I < J < K ≤ N: X(I)+X(J)+X(K), range [0,2]
    c_data = [(i = I, j = J, k = K)
              for I in 1:N for J in I+1:N for K in J+1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = zeros(T, N), uvar = ones(T, N), start = fill(T(2), N))
    ExaModels.@add_obj(c, x[d.i] + d.p * x[d.i]^2 for d in obj_data)
    ExaModels.@add_con(c, x[d.i] - x[d.ip1] for d in e_data; lcon = T(0), ucon = T(2))
    ExaModels.@add_con(c, x[d.i] + x[d.j] + x[d.k] for d in c_data; lcon = T(0), ucon = T(2))
    return ExaModels.ExaModel(c; kwargs...)
end
