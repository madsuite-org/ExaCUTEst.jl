# HAGER3: Hager P3 optimal control (Hager 1990)
# N=2500 discretization points; 2N+1 total vars with X(0) fixed at 1
#
# SIF: FR DEFAULT + XX X(0)=1.0 → X(0) reordered to end
# CUTEst ordering: X(1..N) at pos 1..N, U(1..N) at pos N+1..2N, X(0) at pos 2N+1
#
# LINSQ: XA^2+XA*XB+XB^2; LINU (IV: XX=XA+XB, UU=U): XX*UU=(XA+XB)*U
# OX(I): identity type (SCALE=8/H ignored) → 0.625*LINSQ(X(I-1),X(I)) + LINU(X(I-1),X(I),U(I))
# U(I)SQ: L2 type, SCALE=4/H → U(I)^2*H/4
# Constraints S(I): (1/H-1/4)*X(I) + (-1/H-1/4)*X(I-1) - U(I) = 0
function hager3_model(N = 2500; T = Float64, backend = nothing, kwargs...)
    H = T(1) / T(N)
    invH = T(N)
    coeff1 = invH - T(1) / T(4)   # 1/H - 1/4
    coeff2 = -invH - T(1) / T(4)  # -1/H - 1/4
    nvar = 2 * N + 1
    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    lvar[nvar] = T(1); uvar[nvar] = T(1)   # X(0) fixed at 1
    start = zeros(T, nvar)
    start[nvar] = T(1)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)
    # OX(1): X(0)=x[nvar], X(1)=x[1], U(1)=x[N+1]
    ExaModels.@add_obj(c, T(0.625)*(x[nvar]^2+x[nvar]*x[1]+x[1]^2) + (x[nvar]+x[1])*x[N+1])
    # OX(2..N): X(I-1)=x[i-1], X(I)=x[i], U(I)=x[N+i]
    ExaModels.@add_obj(c, T(0.625)*(x[i-1]^2+x[i-1]*x[i]+x[i]^2) + (x[i-1]+x[i])*x[N+i] for i in 2:N)
    # U(I)SQ
    ExaModels.@add_obj(c, x[N+i]^2 * H / T(4) for i in 1:N)
    # S(1): X(0)=x[nvar]
    ExaModels.@add_con(c, coeff1 * x[1] + coeff2 * x[nvar] - x[N+1])
    # S(2..N)
    ExaModels.@add_con(c, coeff1 * x[i] + coeff2 * x[i-1] - x[N+i] for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end
