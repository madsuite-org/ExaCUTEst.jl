# HAGER3: Hager P3 optimal control (Hager 1990)
# N=2500 discretization points; 2N+1 free vars with X(0) fixed at 1
# Variables: x[1..N+1] = X(0..N), x[N+2..2N+1] = U(1..N)
# OX(I) group: identity type, element = 0.625*(XA^2+XA*XB+XB^2) + (XA+XB)*U
#   where XA=X(I-1), XB=X(I), U=U(I)
# U(I)SQ group: L2 type, SCALE=4/H → contribution = U(I)^2 * H/4
# Constraints (equality): (1/H-1/4)*X(I) + (-1/H-1/4)*X(I-1) - U(I) = 0
function hager3_model(N = 2500; T = Float64, backend = nothing, kwargs...)
    H = T(1) / T(N)
    invH = T(N)
    coeff1 = invH - T(1) / T(4)   # 1/H - 1/4
    coeff2 = -invH - T(1) / T(4)  # -1/H - 1/4
    nvar = 2 * N + 1
    lvar = T[i == 1 ? T(1) : T(-Inf) for i in 1:nvar]
    uvar = T[i == 1 ? T(1) : T(Inf) for i in 1:nvar]
    start = zeros(T, nvar)
    start[1] = T(1)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_obj(
        c,
        T(0.625) * (x[i]^2 + x[i]*x[i+1] + x[i+1]^2) + (x[i] + x[i+1]) * x[N+1+i]
        for i in 1:N)
    ExaModels.@add_obj(c, x[N+1+i]^2 * H / T(4) for i in 1:N)
    ExaModels.@add_con(c, coeff1 * x[i+1] + coeff2 * x[i] - x[N+1+i] for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
