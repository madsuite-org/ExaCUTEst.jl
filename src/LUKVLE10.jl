# LUKVLE10: Generalized Brown function with Broyden tridiagonal constraints
# Source: Problem 5.10, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), N/2=5000 pairs, N-2=9998 equality constraints
# All variables free; start x[odd]=-1, x[even]=1
#
# Objective:
#   sum_{i=1}^{N/2} (x[2i-1]^2)^(x[2i]^2+1)     (OBJ1, identity group)
#   + sum_{i=1}^{N/2} (x[2i]^2)^(x[2i-1]^2+1)   (OBJ2, identity group)
#
# ELEMENT NASTY(X,Y): F = (X^2)^(Y^2+1)
#
# Constraints C(k) for k=1..N-2 (equality = 0):
#   3*x[k+1] - x[k] - 2*x[k+2] - 2*x[k+1]^2 + 1 = 0
#   (SQR element on x[k+1] with coeff -2, rhs=-1)
function lukvle10_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert iseven(N) "N must be even"
    N2 = N ÷ 2

    # Start: odd indices -> -1, even indices -> 1
    start = T[isodd(i) ? T(-1) : T(1) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: NASTY(x[2i-1], x[2i]) + NASTY(x[2i], x[2i-1])
    # NASTY(X,Y) = (X^2)^(Y^2+1)
    ExaModels.@add_obj(c, (x[2i-1]^2)^(x[2i]^2 + T(1)) for i in 1:N2)
    ExaModels.@add_obj(c, (x[2i]^2)^(x[2i-1]^2 + T(1)) for i in 1:N2)

    # Constraints C(k): 3*x[k+1] - x[k] - 2*x[k+2] - 2*x[k+1]^2 + 1 = 0
    # (group linear: 3*x[k+1]-x[k]-2*x[k+2], element SQR(x[k+1])*(-2), rhs=-1)
    ExaModels.@add_con(
        c,
        T(3) * x[k+1] - x[k] - T(2) * x[k+2] - T(2) * x[k+1]^2 + T(1) for k in 1:N-2
    )

    return ExaModels.ExaModel(c; kwargs...)
end
