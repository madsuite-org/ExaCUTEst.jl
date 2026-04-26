# ARTIF: Artificial nonlinear system (Irani et al. 1991)
# N=5000 variables X(1..N) free, X(0) and X(N+1) fixed at 0 (boundaries)
# All variables start at 1 (XV DEFAULT 1.0).
#
# CUTEst variable ordering: the SIF VARIABLES loop starts from I=0 (DO I 0 N+1).
# CUTEst rotates free variables as: x[1]=X(N), x[I+1]=X(I) for I=1..N-1.
# Formula: X(I) → Julia index mod(I,N)+1  (X(N)→1, X(1)→2, ..., X(N-1)→N)
# Fixed: x[N+1]=X(0), x[N+2]=X(N+1).
#
# Constraints EQ(I) for I=1..N:
#   atan(sin(mod(I,100)*X(I))) - 0.05*X(I-1) - 0.05*X(I) - 0.05*X(I+1) = 0
function artif_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    fact = T[T(mod(i, 100)) for i in 1:N]

    lvar = fill(T(-Inf), N + 2)
    uvar = fill(T(Inf), N + 2)
    lvar[N + 1] = uvar[N + 1] = T(0)  # X(0) fixed
    lvar[N + 2] = uvar[N + 2] = T(0)  # X(N+1) fixed

    start = fill(T(1), N + 2)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N + 2; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, FACT, fact)

    # CUTEst rotation: X(I) = x[mod(I,N)+1], X(0)=x[N+1], X(N+1)=x[N+2]
    # EQ(1): X(1)=x[2], X(0)=x[N+1], X(2)=x[3]; FACT=mod(1,100)=1
    ExaModels.@add_con(c,
        atan(sin(T(1) * x[2])) - T(0.05) * x[N + 1] - T(0.05) * x[2] - T(0.05) * x[3])

    # EQ(2..N-2): X(i)=x[i+1], X(i-1)=x[i], X(i+1)=x[i+2]; FACT[i]=mod(i,100)
    ExaModels.@add_con(c,
        atan(sin(FACT[i] * x[i + 1])) - T(0.05) * x[i] - T(0.05) * x[i + 1] -
        T(0.05) * x[i + 2] for i in 2:N-2)

    # EQ(N-1): X(N-1)=x[N], X(N-2)=x[N-1], X(N)=x[1]; FACT=mod(N-1,100)
    ExaModels.@add_con(c,
        atan(sin(T(mod(N - 1, 100)) * x[N])) - T(0.05) * x[N - 1] - T(0.05) * x[N] -
        T(0.05) * x[1])

    # EQ(N): X(N)=x[1], X(N-1)=x[N], X(N+1)=x[N+2]; FACT=mod(N,100)
    ExaModels.@add_con(c,
        atan(sin(T(mod(N, 100)) * x[1])) - T(0.05) * x[N] - T(0.05) * x[1] -
        T(0.05) * x[N + 2])

    return ExaModels.ExaModel(c; kwargs...)
end
