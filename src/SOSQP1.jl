# SOSQP1: Non-convex QP, always second-order sufficient (N=2500 default)
# min  sum_{i=1}^N x_i * y_i
# s.t. x_i - y_i = 1,  i=1..N
#      sum_i (x_i + y_i) = N
#      -1 <= x_i, y_i <= 1
# Variables ordered: x(1), y(1), x(2), y(2), ...
function sosqp1_model(N = 2500; T = Float64, backend = nothing, kwargs...)
    data = [(i = i,) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # variables: x(1..N), y(1..N) interleaved: x[2i-1]=X(i), x[2i]=Y(i)
    ExaModels.@add_var(c, x, 2 * N; lvar = T(-1), uvar = T(1))
    # objective: sum x_i * y_i
    ExaModels.@add_obj(c, x[2 * d.i - 1] * x[2 * d.i] for d in data)
    # equality constraints: X(i) - Y(i) = 1
    ExaModels.@add_con(c, x[2 * d.i - 1] - x[2 * d.i] - T(1) for d in data)
    # one more equality: sum_i (x_i + y_i) = N
    ExaModels.@add_con(c, sum(x[2 * i - 1] + x[2 * i] for i in 1:N) - T(N))
    return ExaModels.ExaModel(c; kwargs...)
end
