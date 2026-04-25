# LUKVLE7: Trigonometric tridiagonal function with simplified five-diagonal constraints
# Source: Problem 5.7, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), 4 equality constraints
# All variables free; start x[i]=1
#
# Objective (single identity group, no group type):
#   sum_{i=1}^{N} (-i)*cos(x[i]) + 2*sum_{i=1}^{N-1} sin(x[i]) - (N-1)*sin(x[N]) + N*(N+1)/2
#
# Constraints (equality = 0):
#   C(1): 4*x[1]+x[2] - 4*x[2]^2 - x[3]^2 = 0
#   C(2): 6*x[2]+x[3] + 8*(x[2]^3-x[2]*x[1]) - 4*x[3]^2 - x[4]^2 - 2 = 0
#   C(3): 6*x[N-1]-x[N-3] + 8*(x[N-1]^3-x[N-1]*x[N-2]) - 4*x[N]^2 + x[N-2]^2 - 2 = 0
#   C(4): 2*x[N]-x[N-2] + 8*(x[N]^3-x[N]*x[N-1]) + x[N-1]^2 = 0
#
# ELEMENT TYPE CUBEP: V^3 - V*W
# ELEMENT TYPE SQR: V^2
function lukvle7_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1))

    # Objective: sum_{i=1}^{N} (-i)*cos(x[i]) + 2*sum_{i=1}^{N-1} sin(x[i]) - (N-1)*sin(x[N])
    #            + N*(N+1)/2
    # (Collected from SIF GROUP USES: CO(i) coeff=-i for all i;
    #  SI(i) coeff=2 for i=1..N-1, coeff=-(N-1) for i=N)
    ExaModels.@add_par(c, ipar, T[T(i) for i in 1:N])
    ExaModels.@add_obj(c, -ipar[i] * cos(x[i]) for i in 1:N)
    ExaModels.@add_obj(c, T(2) * sin(x[i]) for i in 1:N-1)
    ExaModels.@add_obj(c, -T(N - 1) * sin(x[N]))
    ExaModels.@add_obj(c, T(N) * T(N + 1) / T(2))

    # Constraints:
    # C(1): 4*x[1]+x[2] - 4*x[2]^2 - x[3]^2 = 0
    ExaModels.@add_con(c, T(4) * x[1] + x[2] - T(4) * x[2]^2 - x[3]^2)

    # C(2): 6*x[2]+x[3] + 8*(x[2]^3-x[2]*x[1]) - 4*x[3]^2 - x[4]^2 - 2 = 0
    ExaModels.@add_con(
        c,
        T(6) * x[2] + x[3] + T(8) * (x[2]^3 - x[2] * x[1]) - T(4) * x[3]^2 - x[4]^2 -
        T(2),
    )

    # C(3): 6*x[N-1]-x[N-3] + 8*(x[N-1]^3-x[N-1]*x[N-2]) - 4*x[N]^2 + x[N-2]^2 - 2 = 0
    ExaModels.@add_con(
        c,
        T(6) * x[N-1] - x[N-3] + T(8) * (x[N-1]^3 - x[N-1] * x[N-2]) - T(4) * x[N]^2 +
        x[N-2]^2 - T(2),
    )

    # C(4): 2*x[N]-x[N-2] + 8*(x[N]^3-x[N]*x[N-1]) + x[N-1]^2 = 0
    ExaModels.@add_con(
        c,
        T(2) * x[N] - x[N-2] + T(8) * (x[N]^3 - x[N] * x[N-1]) + x[N-1]^2,
    )

    return ExaModels.ExaModel(c; kwargs...)
end
