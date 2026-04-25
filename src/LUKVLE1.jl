# LUKVLE1: Chained Rosenbrock function with simplified trigonometric exponential constraints
# Source: Problem 5.1, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), N-2=9998 equality constraints
# All variables free; start: odd→-1.2, even→1.0
#
# Objective (Rosenbrock, no variable reordering):
#   sum_{I=1}^{N-1} [100*(X(I)^2 - X(I+1))^2 + (X(I) - 1)^2]
#   = sum_{I=1}^{N-1} [(X(I)^2 - X(I+1))^2 / 0.01 + (X(I) - 1)^2]
#
# Constraints C(K) for K=1..N-2 (identity group type):
#   3*X(K+1)^3 + sin(X(K+1)-X(K+2))*sin(X(K+1)+X(K+2))
#   - X(K)*exp(X(K)-X(K+1)) + 2*X(K+2) + 4*X(K+1) - 8 = 0
#
# No variable reordering (CUTEst pos = SIF index for all variables).
function lukvle1_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    start = T[isodd(i) ? T(-1.2) : T(1.0) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: L2 groups Q(I) and L(I) for I=1..N-1
    # Q(I): SCALE=0.01, v = -X(I+1) + X(I)^2; contrib = v^2/0.01 = 100*v^2
    # L(I): v = X(I) - 1; contrib = v^2
    ExaModels.@add_obj(
        c,
        T(100) * (-x[i+1] + x[i]^2)^2 + (x[i] - T(1))^2 for i in 1:(N-1)
    )

    # Constraints C(K): elements CUBE(X(K+1)) coeff 3, SINASINB(X(K+1),X(K+2)) coeff 1,
    #   XEXP(X(K),X(K+1)) coeff -1; linear 2*X(K+2) + 4*X(K+1); rhs=8
    # SINASINB: sin(X1-X2)*sin(X1+X2)
    # XEXP: X1*exp(X1-X2)
    ExaModels.@add_con(
        c,
        T(3) * x[k+1]^3 + sin(x[k+1] - x[k+2]) * sin(x[k+1] + x[k+2]) -
        x[k] * exp(x[k] - x[k+1]) + T(2) * x[k+2] + T(4) * x[k+1] - T(8)
        for k in 1:(N-2)
    )

    return ExaModels.ExaModel(c; kwargs...)
end
