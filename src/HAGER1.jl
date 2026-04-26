# HAGER1: Hager P1 optimal control (Hager 1990)
# N=2500 discretization points; 2N+1 total vars with X(0) fixed at 1
#
# SIF: FR DEFAULT + XX X(0)=1.0 → X(0) reordered to end
# CUTEst ordering: X(1..N) at pos 1..N, U(1..N) at pos N+1..2N, X(0) at pos 2N+1
#
# Objective: X(N)^2/2 + sum_{I=1..N} U(I)^2*H/2
# Constraints S(I): (1/H-1/2)*X(I) + (-1/H-1/2)*X(I-1) - U(I) = 0
function hager1_model(N = 2500; T = Float64, backend = nothing, kwargs...)
    H = T(1) / T(N)
    invH = T(N)
    coeff1 = invH - T(1) / T(2)   # 1/H - 1/2
    coeff2 = -invH - T(1) / T(2)  # -1/H - 1/2
    nvar = 2 * N + 1
    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    lvar[nvar] = T(1); uvar[nvar] = T(1)   # X(0) fixed at 1
    start = zeros(T, nvar)
    start[nvar] = T(1)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_obj(c, x[N]^2 / T(2))
    ExaModels.@add_obj(c, x[N+i]^2 * H / T(2) for i in 1:N)
    # S(1): X(0)=x[nvar], X(1)=x[1], U(1)=x[N+1]
    ExaModels.@add_con(c, coeff1 * x[1] + coeff2 * x[nvar] - x[N+1])
    # S(2..N)
    ExaModels.@add_con(c, coeff1 * x[i] + coeff2 * x[i-1] - x[N+i] for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end
