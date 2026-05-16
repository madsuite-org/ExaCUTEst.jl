# SOSQP2: Non-convex QP, always second-order sufficient (N=2500 default)
# min  sum_{i=1}^N x_i * y_i
# s.t. i*x_i - y_i = i,  i=1..N
#      sum_i (x_i + y_i) = N/2
#      -1 <= x_i, y_i <= 1
function sosqp2_model(N = 2500; T = Float64, backend = nothing, kwargs...)
    data = [(i = i, ri = T(i)) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2 * N; lvar = T(-1), uvar = T(1))
    ExaModels.@add_obj(c, x[2 * d.i - 1] * x[2 * d.i] for d in data)
    ExaModels.@add_con(c, d.ri * x[2 * d.i - 1] - x[2 * d.i] - d.ri for d in data)
    # sum constraint: sum(x_i + y_i) = N/2 — use @add_con!
    g = ExaModels.@add_con(c, x[1] + x[2] - T(N) / T(2))
    ExaModels.@add_con!(c, g[1] += x[2 * d.i - 1] + x[2 * d.i] for d in data[2:end])
    return ExaModels.ExaModel(c; kwargs...)
end
