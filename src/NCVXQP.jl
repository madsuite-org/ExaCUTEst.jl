# NCVXQP1-9: Non-convex quadratic programs (N=10000 default)
# Objective: sum_{I=1}^NPLUS 0.5*I*(v_I)^2 - sum_{I=NPLUS+1}^N 0.5*I*(v_I)^2
#   where v_I = X(I) + X(mod(2I-1,N)+1) + X(mod(3I-1,N)+1)
# Constraints: X(I) + 2*X(mod(4I-1,N)+1) + 3*X(mod(5I-1,N)+1) = 6 for I=1..M
# Variants differ in NPLUS (fraction of positive terms) and M (number of constraints)

function _ncvxqp_model(N, NPLUS, M; T = Float64, backend = nothing, kwargs...)
    data_pos = [(i = i, j2 = mod(2*i-1, N)+1, j3 = mod(3*i-1, N)+1, p = T(0.5*i)) for i in 1:NPLUS]
    data_neg = [(i = i, j2 = mod(2*i-1, N)+1, j3 = mod(3*i-1, N)+1, p = T(-0.5*i)) for i in NPLUS+1:N]
    data_con = [(i = i, j4 = mod(4*i-1, N)+1, j5 = mod(5*i-1, N)+1) for i in 1:M]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0.1), uvar = T(10.0), start = T(0.5))
    ExaModels.@add_obj(c,
        d.p * (x[d.i] + x[d.j2] + x[d.j3])^2 for d in data_pos)
    ExaModels.@add_obj(c,
        d.p * (x[d.i] + x[d.j2] + x[d.j3])^2 for d in data_neg)
    ExaModels.@add_con(c,
        x[d.i] + T(2)*x[d.j4] + T(3)*x[d.j5] - T(6) for d in data_con)
    return ExaModels.ExaModel(c; kwargs...)
end

ncvxqp1_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxqp_model(N, N÷4,   N÷2;   T = T, backend = backend, kwargs...)
ncvxqp2_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxqp_model(N, N÷2,   N÷2;   T = T, backend = backend, kwargs...)
ncvxqp3_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxqp_model(N, 3*(N÷4), N÷2; T = T, backend = backend, kwargs...)
ncvxqp4_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxqp_model(N, N÷4,   N÷4;   T = T, backend = backend, kwargs...)
ncvxqp5_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxqp_model(N, N÷2,   N÷4;   T = T, backend = backend, kwargs...)
ncvxqp6_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxqp_model(N, 3*(N÷4), N÷4; T = T, backend = backend, kwargs...)
ncvxqp7_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxqp_model(N, N÷4,   3*(N÷4); T = T, backend = backend, kwargs...)
ncvxqp8_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxqp_model(N, N÷2,   3*(N÷4); T = T, backend = backend, kwargs...)
ncvxqp9_model(N = 10000; T = Float64, backend = nothing, kwargs...) =
    _ncvxqp_model(N, 3*(N÷4), 3*(N÷4); T = T, backend = backend, kwargs...)
