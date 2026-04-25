# LUKVLE9: Modified Brown function with simplified seven-diagonal constraints
# Source: Problem 5.9, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), 6 equality constraints
# All variables free; start x[i]=-1
#
# Objective:
#   sum_{i=1}^{N/2} 0.001*(x[2i-1])^2                  (OBJ1, AL2 group type)
#   + sum_{i=1}^{N/2} (-x[2i-1]+x[2i])                 (OBJ2, identity group, scalar)
#   + sum_{i=1}^{N/2} exp(20*(x[2i-1]-x[2i]))           (OBJ3, EXP20 group type)
#
# GROUP TYPE AL2: F(v) = 0.001*v^2
# GROUP TYPE EXP20: F(v) = exp(20*v)
# ELEMENT CUBEP(v,w): v^3 - v*w
# ELEMENT SQR(v): v^2
#
# Constraints (equality = 0):
# C(1): 4*x[1]+x[2]+x[3] - 4*x[2]^2 - x[3]^2 - x[4]^2 = 0
# C(2): 6*x[2]+x[3]+x[4] + 8*(x[2]^3-x[2]*x[1]) - 4*x[3]^2 + x[1]^2 - x[4]^2 - x[5]^2 - 2 = 0
# C(3): 6*x[3]+x[4]+x[5]-x[1] + 8*(x[3]^3-x[3]*x[2]) - 4*x[4]^2 + x[2]^2 - x[5]^2 + x[1]^2 - x[6]^2 - 2 = 0
# C(4): 6*x[N-2]+x[N-1]+x[N]-x[N-4]-x[N-5] + 8*(x[N-2]^3-x[N-2]*x[N-3]) - 4*x[N-1]^2 + x[N-3]^2 - x[N]^2 + x[N-4]^2 - 2 = 0
# C(5): 6*x[N-1]-x[N-3]+x[N]-x[N-4] + 8*(x[N-1]^3-x[N-1]*x[N-2]) - 4*x[N]^2 + x[N-2]^2 + x[N-3]^2 - 2 = 0
# C(6): 2*x[N]-x[N-3]-x[N-2] + 8*(x[N]^3-x[N]*x[N-1]) + x[N-1]^2 + x[N-2]^2 = 0
function lukvle9_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert iseven(N) "N must be even"
    N2 = N ÷ 2

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(-1))

    # Objective: OBJ1 group terms (AL2: 0.001*v^2), v = x[2i-1]
    ExaModels.@add_obj(c, T(0.001) * x[2i-1]^2 for i in 1:N2)

    # OBJ2 group (identity, scalar): v = sum_i (-x[2i-1]+x[2i])
    ExaModels.@add_obj(c, -x[2i-1] + x[2i] for i in 1:N2)

    # OBJ3 group terms (EXP20: exp(20*v)), v = x[2i-1]-x[2i]
    ExaModels.@add_obj(c, exp(T(20) * (x[2i-1] - x[2i])) for i in 1:N2)

    # Constraints:
    # C(1)
    ExaModels.@add_con(
        c,
        T(4) * x[1] + x[2] + x[3] - T(4) * x[2]^2 - x[3]^2 - x[4]^2,
    )
    # C(2)
    ExaModels.@add_con(
        c,
        T(6) * x[2] + x[3] + x[4] + T(8) * (x[2]^3 - x[2] * x[1]) - T(4) * x[3]^2 +
        x[1]^2 - x[4]^2 - x[5]^2 - T(2),
    )
    # C(3)
    ExaModels.@add_con(
        c,
        T(6) * x[3] + x[4] + x[5] - x[1] + T(8) * (x[3]^3 - x[3] * x[2]) -
        T(4) * x[4]^2 + x[2]^2 - x[5]^2 + x[1]^2 - x[6]^2 - T(2),
    )
    # C(4)
    ExaModels.@add_con(
        c,
        T(6) * x[N-2] + x[N-1] + x[N] - x[N-4] - x[N-5] +
        T(8) * (x[N-2]^3 - x[N-2] * x[N-3]) - T(4) * x[N-1]^2 + x[N-3]^2 - x[N]^2 +
        x[N-4]^2 - T(2),
    )
    # C(5)
    ExaModels.@add_con(
        c,
        T(6) * x[N-1] - x[N-3] + x[N] - x[N-4] +
        T(8) * (x[N-1]^3 - x[N-1] * x[N-2]) - T(4) * x[N]^2 + x[N-2]^2 + x[N-3]^2 -
        T(2),
    )
    # C(6)
    ExaModels.@add_con(
        c,
        T(2) * x[N] - x[N-3] - x[N-2] + T(8) * (x[N]^3 - x[N] * x[N-1]) + x[N-1]^2 +
        x[N-2]^2,
    )

    return ExaModels.ExaModel(c; kwargs...)
end
