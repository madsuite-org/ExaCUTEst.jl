# LUKVLE8: Augmented Lagrangian function with discrete boundary value constraints
# Source: Problem 5.8, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), N/5=2000 objective groups, N-2=9998 equality constraints
# All variables free; start x[odd]=-1, x[even]=2
#
# Parameters:
#   LAMBDA1=-0.002008, LAMBDA2=-0.001900, LAMBDA3=-0.000261
#   H=1/(N+1), H^2/2 = 0.5/((N+1)^2)
#   LAMBA1+10 = LAMBDA1+10 = 9.997992
#   LAMBA3-1 = LAMBDA3-1 = -1.000261
#
# Objective terms (block i=1..N/5, 5 vars per block: x[5i-4..5i]):
#   O1(i): exp(x[5i]*x[5i-1]*x[5i-2]*x[5i-3]*x[5i-4])
#   O2(i): 10*(x[5i]^2+x[5i-1]^2+...+x[5i-4]^2 - (LAMBDA1+10))^2
#   O3(i): 10*(x[5i-3]*x[5i-2] - 5*x[5i-1]*x[5i] - LAMBDA2)^2
#   O4(i): 10*(x[5i-4]^3+x[5i-3]^3 - (LAMBDA3-1))^2
#
# Constraints C(k) for k=1..N-2 (equality = 0):
#   2*x[k+1] - x[k] - x[k+2] + H^2/2*(x[k+1] + h*(k+1)+1)^2 = 0
#   where h = H = 1/(N+1), so h*(k+1)+1 = (k+1)/(N+1)+1
function lukvle8_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert N % 5 == 0 "N must be divisible by 5"
    N5 = N ÷ 5
    N2 = N - 2

    lambda1 = T(-0.002008)
    lambda2 = T(-0.001900)
    lambda3 = T(-0.000261)
    lamba1p10 = lambda1 + T(10)   # LAMBA1+10 = 9.997992
    lamba3m1 = lambda3 - T(1)     # LAMBA3-1 = -1.000261

    h = T(1) / T(N + 1)
    h2h2 = T(0.5) * h^2           # H^2/2

    # Start: odd indices -> -1, even indices -> 2
    start = T[isodd(i) ? T(-1) : T(2) for i in 1:N]

    # Precompute shift values P(k) = h*(k+1)+1 for constraint SHIFTSQR elements
    # P(k+1) = (k+1)*H + 1, used as (x[k+1] + P(k))^2 in constraint C(k)
    P = T[(k + 1) * h + T(1) for k in 1:N2]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    ExaModels.@add_par(c, P, P)

    # Objective: O1 terms (group type EXP: exp(v))
    ExaModels.@add_obj(
        c,
        exp(x[5i] * x[5i-1] * x[5i-2] * x[5i-3] * x[5i-4]) for i in 1:N5
    )

    # O2 terms (group type 10L2: 10*v^2)
    ExaModels.@add_obj(
        c,
        T(10) * (x[5i]^2 + x[5i-1]^2 + x[5i-2]^2 + x[5i-3]^2 + x[5i-4]^2 - lamba1p10)^2
        for i in 1:N5
    )

    # O3 terms (group type 10L2: 10*v^2)
    ExaModels.@add_obj(
        c,
        T(10) * (x[5i-3] * x[5i-2] - T(5) * x[5i-1] * x[5i] - lambda2)^2 for i in 1:N5
    )

    # O4 terms (group type 10L2: 10*v^2)
    ExaModels.@add_obj(
        c,
        T(10) * (x[5i-4]^3 + x[5i-3]^3 - lamba3m1)^2 for i in 1:N5
    )

    # Constraints C(k): 2*x[k+1] - x[k] - x[k+2] + H^2/2*(x[k+1]+P[k])^2 = 0
    ExaModels.@add_con(
        c,
        T(2) * x[k+1] - x[k] - x[k+2] + h2h2 * (x[k+1] + P[k])^2 for k in 1:N2
    )

    return ExaModels.ExaModel(c; kwargs...)
end
