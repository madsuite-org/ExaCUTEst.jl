# HVYCRASH: Heavy spacecraft landing optimal control, N=1000
# 4*(N+1) = 4004 variables, 3*N = 3000 equality constraints
# State: X(1,T), X(2,T), X(3,T) for T=0..N
# Control: U(T) for T=0..N
# Variable ordering: X(1,0..N), X(2,0..N), X(3,0..N), U(0..N)
#
# Problem parameters:
# TT=0.2185, EPS=0.1, ALPHA=1.62079
# H = TT/N = 0.2185/1000
#
# Fixed: X(1,0)=0, X(2,0)=2.19905, X(3,0)=0, X(3,N)=0
# Bounds: U(T) ∈ [0.08, 0.417], X(3,T) ∈ [0, 2π]
# Start: X(2,T)=1.5 for T=1..N-1, X(2,0)=2.19905, X(2,N)=1.09905
#
# Objective: minimize X(1,N) = x[N+1] (last state variable 1)
#
# Elements (with ALPHA=A):
# C = 0.01 + 0.3*UU^2
# C2(XX3,XX2,UU;A) = cos(XX3) / (A*C*XX2^2)
# INV(V) = 1/V
# C3(XX3,XX2,UU;A) = cos(XX3) / (A*C*XX2^3)
# UX2(XX2,UU) = UU / (C*XX2^2)
# C4(XX3,XX2,UU;A) = cos(XX3) / (A*C*XX2^4)
#
# Constraints:
# C(1,T): X(1,T-1) - X(1,T) + H*C2(X(3,T),X(2,T),U(T)) = 0
# C(2,T): -INV(X(2,T)) - C3(X(3,T),X(2,T),U(T)) = 0
# C(3,T): EPS*X(3,T-1) - EPS*X(3,T) + H*UX2(X(2,T),U(T)) - H*C4(X(3,T),X(2,T),U(T)) = 0
function hvycrash_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    TT = T(0.2185)
    EPS = T(0.1)
    ALPHA = T(1.62079)
    H = TT / T(N)

    NP1 = N + 1
    nvar = 4 * NP1
    # x[1..NP1] = X(1,0..N), x[NP1+1..2NP1] = X(2,0..N)
    # x[2NP1+1..3NP1] = X(3,0..N), x[3NP1+1..4NP1] = U(0..N)

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    start = zeros(T, nvar)

    # X(2,T) default start
    for t in 1:(N-1)
        start[NP1 + t + 1] = T(1.5)
    end

    # Fixed: X(1,0)=0, X(2,0)=2.19905, X(3,0)=0
    lvar[1] = T(0);       uvar[1] = T(0);       start[1] = T(0)
    lvar[NP1+1] = T(2.19905); uvar[NP1+1] = T(2.19905); start[NP1+1] = T(2.19905)
    lvar[2*NP1+1] = T(0); uvar[2*NP1+1] = T(0); start[2*NP1+1] = T(0)

    # Fixed: X(3,N)=0
    lvar[3*NP1] = T(0);   uvar[3*NP1] = T(0)

    # X(2,N) start
    start[2*NP1] = T(1.09905)

    # U(T) bounds and X(3,T) bounds for T=0..N
    for t in 0:N
        idx_u = 3 * NP1 + t + 1
        lvar[idx_u] = T(0.08)
        uvar[idx_u] = T(0.417)
        idx_x3 = 2 * NP1 + t + 1
        lvar[idx_x3] = T(0)
        uvar[idx_x3] = T(6.2831854)
    end
    # X(3,0) fixed to 0 already set above
    # X(3,N) fixed to 0 already set above

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Objective: X(1,N) = x[N+1]
    ExaModels.@add_obj(c, x[N + 1])

    # State constraints for T=1..N:
    # Index: X(1,T) = x[T+1], X(2,T) = x[NP1+T+1], X(3,T) = x[2NP1+T+1], U(T) = x[3NP1+T+1]

    # C(1,T): X(1,T-1) - X(1,T) + H * cos(X(3,T)) / (ALPHA*(0.01+0.3*U(T)^2)*X(2,T)^2) = 0
    ExaModels.@add_con(c,
        x[t] - x[t + 1] +
        H * cos(x[2*NP1 + t + 1]) /
            (ALPHA * (T(0.01) + T(0.3) * x[3*NP1 + t + 1]^2) * x[NP1 + t + 1]^2)
        for t in 1:N)

    # C(2,T): -1/X(2,T) - cos(X(3,T)) / (ALPHA*(0.01+0.3*U(T)^2)*X(2,T)^3) = 0
    ExaModels.@add_con(c,
        -T(1) / x[NP1 + t + 1] -
        cos(x[2*NP1 + t + 1]) /
            (ALPHA * (T(0.01) + T(0.3) * x[3*NP1 + t + 1]^2) * x[NP1 + t + 1]^3)
        for t in 1:N)

    # C(3,T): EPS*X(3,T-1) - EPS*X(3,T) + H*U(T)/((0.01+0.3*U(T)^2)*X(2,T)^2)
    #         - H*cos(X(3,T)) / (ALPHA*(0.01+0.3*U(T)^2)*X(2,T)^4) = 0
    ExaModels.@add_con(c,
        EPS * x[2*NP1 + t] - EPS * x[2*NP1 + t + 1] +
        H * x[3*NP1 + t + 1] / ((T(0.01) + T(0.3) * x[3*NP1 + t + 1]^2) * x[NP1 + t + 1]^2) -
        H * cos(x[2*NP1 + t + 1]) /
            (ALPHA * (T(0.01) + T(0.3) * x[3*NP1 + t + 1]^2) * x[NP1 + t + 1]^4)
        for t in 1:N)

    return ExaModels.ExaModel(c; kwargs...)
end
