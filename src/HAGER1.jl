# HAGER1: Hager P1 optimal control (Hager 1990)
# N=2500 discretization points; 2N+1 free vars with X(0) fixed at 1
# Variables: x[1..N+1] = X(0..N), x[N+2..2N+1] = U(1..N)
# Objective: X(N)^2/2 + sum_{I=1..N} U(I)^2 * H/2  (H=1/N)
# Constraints (equality): (1/H-1/2)*X(I) + (-1/H-1/2)*X(I-1) - U(I) = 0
function hager1_model(N = 2500; T = Float64, backend = nothing, kwargs...)
    H = T(1) / T(N)
    invH = T(N)
    coeff1 = invH - T(1) / T(2)   # 1/H - 1/2
    coeff2 = -invH - T(1) / T(2)  # -1/H - 1/2
    nvar = 2 * N + 1
    lvar = T[i == 1 ? T(1) : T(-Inf) for i in 1:nvar]
    uvar = T[i == 1 ? T(1) : T(Inf) for i in 1:nvar]
    start = zeros(T, nvar)
    start[1] = T(1)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_obj(c, x[N+1]^2 / T(2))
    ExaModels.@add_obj(c, x[N+1+i]^2 * H / T(2) for i in 1:N)
    ExaModels.@add_con(c, coeff1 * x[i+1] + coeff2 * x[i] - x[N+1+i] for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
