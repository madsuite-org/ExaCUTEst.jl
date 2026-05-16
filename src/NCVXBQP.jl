# NCVXBQP1/2/3: Non-convex bound-constrained quadratic programs (N=10000 default)
# Group type SQR with parameter P: F = 0.5*P*alpha^2
# objective = sum_i 0.5*i*(x[i]+x[j1]+x[j2])^2 for i=1..NPLUS (positive)
#           + sum_i 0.5*(-i)*(x[i]+x[j1]+x[j2])^2 for i=NPLUS+1..N (negative)
# j1 = mod(2i-1,N)+1, j2 = mod(3i-1,N)+1
# Bounds: 0.1 <= x[i] <= 10

function _ncvxbqp_model(N, NPLUS; T = Float64, backend = nothing, kwargs...)
    j1(i) = mod(2i - 1, N) + 1
    j2(i) = mod(3i - 1, N) + 1
    data_pos = [(i = i, j1 = j1(i), j2 = j2(i), p = T(0.5 * i))  for i in 1:NPLUS]
    data_neg = [(i = i, j1 = j1(i), j2 = j2(i), p = T(-0.5 * i)) for i in NPLUS+1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0.1), uvar = T(10.0), start = T(0.5))
    ExaModels.@add_obj(c, d.p * (x[d.i] + x[d.j1] + x[d.j2])^2 for d in data_pos)
    ExaModels.@add_obj(c, d.p * (x[d.i] + x[d.j1] + x[d.j2])^2 for d in data_neg)
    return ExaModels.ExaModel(c; kwargs...)
end

ncvxbqp1_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxbqp_model(N, N ÷ 4; T = T, backend = backend, kwargs...)
ncvxbqp2_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxbqp_model(N, N ÷ 2; T = T, backend = backend, kwargs...)
ncvxbqp3_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxbqp_model(N, 3 * (N ÷ 4); T = T, backend = backend, kwargs...)
