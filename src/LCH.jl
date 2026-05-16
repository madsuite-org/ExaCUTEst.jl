# LCH: Minimum energy problem in quantum physics (3N variables, 1 constraint)
# Source: Leonor Cruzeiro-Hasson, via Nick Gould, Feb 1991.
# classification OOR2-MY-V-1
# N=1000 (default), 3000 total variables
# Variables: PR(i), PI(i), U(i) for i=1..N (periodic: index 0 wraps to N)
# x[i]=PR(i), x[N+i]=PI(i), x[2N+i]=U(i) for i=1..N
# Objective: sum_i [ -J*PR(i)*(PR(i-1)+PR(i+1)) - J*PI(i)*(PI(i-1)+PI(i+1))
#                  + K*(U(i-1)-U(i))^2 + (A*U(i-1)+B*U(i)+C*U(i+1))*(PR(i)^2+PI(i)^2) ]
# Constraint CON = 1: sum_i (PR(i)^2 + PI(i)^2) = 1
# Constants: J=1.549429, K=650.0, X1=62.0, X2=62.0
# A = -X1 = -62.0, B = X1-X2 = 0.0, C = X2 = 62.0
# STARTP = 1/sqrt(2N), U(i) starts at 0.1*i
function lch_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    J = T(1.549429)
    K = T(650.0)
    A = T(-62.0)
    C_val = T(62.0)
    startp = T(1.0 / sqrt(2 * N))
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # All variables free, arranged as PR(1..N), PI(1..N), U(1..N)
    start_vals = vcat(fill(startp, N), fill(startp, N), T[0.1 * i for i in 1:N])
    ExaModels.@add_var(c, x, 3 * N; lvar = fill(T(-Inf), 3 * N), uvar = fill(T(Inf), 3 * N),
                       start = start_vals)
    # Objective: sum over i=1..N
    # Periodic indices: PR(i-1) = x[mod(i-2,N)+1], PR(i+1) = x[mod(i,N)+1]
    # PI(i-1) = x[N + mod(i-2,N)+1], PI(i+1) = x[N + mod(i,N)+1]
    # U(i-1)  = x[2N + mod(i-2,N)+1], U(i+1)  = x[2N + mod(i,N)+1]
    # Term 1: -J * PR(i) * (PR(i-1) + PR(i+1))
    ExaModels.@add_obj(c, -J * x[i] * (x[mod(i - 2, N) + 1] + x[mod(i, N) + 1])
                       for i in 1:N)
    # Term 2: -J * PI(i) * (PI(i-1) + PI(i+1))
    ExaModels.@add_obj(c, -J * x[N + i] * (x[N + mod(i - 2, N) + 1] + x[N + mod(i, N) + 1])
                       for i in 1:N)
    # Term 3: K * (U(i-1) - U(i))^2
    ExaModels.@add_obj(c, K * (x[2 * N + mod(i - 2, N) + 1] - x[2 * N + i])^2
                       for i in 1:N)
    # Term 4: (A*U(i-1) + C*U(i+1)) * (PR(i)^2 + PI(i)^2)
    # B=0 so middle term vanishes
    ExaModels.@add_obj(c,
                       (A * x[2 * N + mod(i - 2, N) + 1] + C_val * x[2 * N + mod(i, N) + 1]) *
                       (x[i]^2 + x[N + i]^2)
                       for i in 1:N)
    # Constraint: sum(PR(i)^2 + PI(i)^2) = 1  (rhs=1 → expr - 1 = 0)
    ExaModels.@add_con(c, sum(x[i]^2 + x[N + i]^2 for i in 1:N) - T(1.0))
    return ExaModels.ExaModel(c; kwargs...)
end
