function cvxqp2_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    M = N ÷ 4
    j1(i) = mod(2i - 1, N) + 1
    j2(i) = mod(3i - 1, N) + 1
    j3(i) = mod(4i - 1, N) + 1
    j4(i) = mod(5i - 1, N) + 1
    obj_data = [(i = i, j1 = j1(i), j2 = j2(i), p = T(0.5 * i)) for i in 1:N]
    con_data = [(i = i, j3 = j3(i), j4 = j4(i)) for i in 1:M]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(T(0.1), N), uvar = fill(T(10), N), start = fill(T(0.5), N))
    ExaModels.@add_obj(c, d.p * (x[d.i] + x[d.j1] + x[d.j2])^2 for d in obj_data)
    ExaModels.@add_con(c, x[d.i] + T(2) * x[d.j3] + T(3) * x[d.j4] - T(6) for d in con_data)
    return ExaModels.ExaModel(c; kwargs...)
end
