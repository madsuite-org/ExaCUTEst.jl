function cvxbqp1_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    j1(i) = mod(2i - 1, N) + 1
    j2(i) = mod(3i - 1, N) + 1
    data = [(i = i, j1 = j1(i), j2 = j2(i), p = T(0.5 * i)) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(T(0.1), N), uvar = fill(T(10), N), start = fill(T(0.5), N))
    ExaModels.@add_obj(c, d.p * (x[d.i] + x[d.j1] + x[d.j2])^2 for d in data)
    return ExaModels.ExaModel(c; kwargs...)
end
