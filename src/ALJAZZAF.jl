# ALJAZZAF: Quadratic nonlinear program from Aljazzaf (1990)
# N=1000 variables (default), N1=500
# Default lower bound x >= 0 (no BOUNDS section)
# CUTEst variable ordering: position 1 = X(N), positions 2..N-1 = X(2..N-1), position N = X(1)
# A(I) = 100 - (I-1)*(A^2-1)/((N-1)*A), B(I) = (I-1)*(A^2-1)/(N-1) + 1, A=100
# Source: M. Aljazzaf PhD thesis, NC State, 1990
function aljazzaf_model(N = 1000, N1 = 500; T = Float64, backend = nothing, kwargs...)
    BIGA  = T(100)
    ASQ1  = BIGA * BIGA - T(1)
    RNm1  = T(N - 1)
    F     = ASQ1 / RNm1
    F2    = F / BIGA

    Acoef = T[BIGA - T(i - 1) * F2 for i in 1:N]
    Bcoef = T[T(i - 1) * F + T(1)  for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0), uvar = T(Inf), start = T(0))

    # CUTEst ordering: x[1]=X(N), x[2..N-1]=X(2..N-1), x[N]=X(1)
    # Objective: I=N (X(N)→x[1], shift=1): A(N)*(x[1]-1)^2
    ExaModels.@add_obj(c, Acoef[N] * (x[1] - T(1))^2)
    # Objective: I=2..N1 (X(2..N1)→x[2..N1], shift=-1): A(I)*(x[I]+1)^2
    ExaModels.@add_par(c, Apar1, Acoef[2:N1])
    ExaModels.@add_obj(c, Apar1[k] * (x[k + 1] + T(1))^2 for k in 1:N1-1)
    # Objective: I=N1+1..N-1 (X(N1+1..N-1)→x[N1+1..N-1], shift=1)
    ExaModels.@add_par(c, Apar2, Acoef[N1+1:N-1])
    ExaModels.@add_obj(c, Apar2[k] * (x[N1 + k] - T(1))^2 for k in 1:N-N1-1)
    # Objective: I=1 (X(1)→x[N], shift=0.5): A(1)*(x[N]-0.5)^2
    ExaModels.@add_obj(c, Acoef[1] * (x[N] - T(0.5))^2)

    # Constraint H = 0:
    # -B(1)*X(1) + B(1) + sum_{I=2}^{N1} B(I)*X(I)^2 + sum_{I=N1+1}^{N} B(I)*(X(I)-1)^2
    # X(1)→x[N], X(2..N-1)→x[2..N-1], X(N)→x[1]
    Bcoef1 = Bcoef[2:N1]
    Bcoef2 = Bcoef[N1+1:N-1]
    ExaModels.@add_con(c,
        -Bcoef[1] * x[N] + Bcoef[1]
        + sum(Bcoef1[k] * x[k + 1]^2 for k in 1:N1-1)
        + sum(Bcoef2[k] * (x[N1 + k] - T(1))^2 for k in 1:N-N1-1)
        + Bcoef[N] * (x[1] - T(1))^2)

    return ExaModels.ExaModel(c; kwargs...)
end
