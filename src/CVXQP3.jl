# CVXQP3: Convex quadratic program (N=10000 variables, M=3*N/4 constraints)
# Objective: sum_{i=1}^N  0.5 * i * (x_i + x_{j2(i)} + x_{j3(i)})^2
#   where j2(i) = mod(2i-1,N)+1, j3(i) = mod(3i-1,N)+1
#   (GROUP TYPE SQR with param P=i: F = 0.5*P*ALPHA^2)
# Constraints: x_i + 2*x_{j4(i)} + 3*x_{j5(i)} = 6  for i=1..M
#   where j4(i) = mod(4i-1,N)+1, j5(i) = mod(5i-1,N)+1, M = 3*(N/4)
# Bounds: x_i in [0.1, 10.0]; start = 0.5
function cvxqp3_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    M = 3 * (N ÷ 4)
    data_obj = [(i = i, j2 = mod(2 * i - 1, N) + 1, j3 = mod(3 * i - 1, N) + 1, p = T(0.5 * i)) for i in 1:N]
    data_con = [(i = i, j4 = mod(4 * i - 1, N) + 1, j5 = mod(5 * i - 1, N) + 1) for i in 1:M]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0.1), uvar = T(10.0), start = T(0.5))
    ExaModels.@add_obj(c,
        d.p * (x[d.i] + x[d.j2] + x[d.j3])^2 for d in data_obj)
    ExaModels.@add_con(c,
        x[d.i] + T(2) * x[d.j4] + T(3) * x[d.j5] - T(6) for d in data_con)
    return ExaModels.ExaModel(c; kwargs...)
end
