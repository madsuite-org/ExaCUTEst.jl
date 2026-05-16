# SSBRYBND: Scaled Broyden banded system in least-squares sense
# NB: scaled version of BRYBND with SCALE(I) = exp((I-1)/(N-1)*6)
# N=5000 (default), LB=5, UB=1, KAPPA1=2, KAPPA2=5, KAPPA3=1
# SQ element: F(V,P) = P^2 * V^2
# CB element: F(V,P) = P^3 * V^3
# Group type L2: contribution = (gvar)^2
#
# Upper-left (I=1..LB): diagonal uses CB, all neighbors use SQ
#   v[I] = K1*s_i*x_i + K2*s_i^3*x_i^3 - K3*sum_j(s_j*x_j + s_j^2*x_j^2)
# Main (I=LB+1..N-UB-1): lower-band uses CB, diagonal & upper-band use SQ
#   v[I] = K1*s_i*x_i + K2*s_i^2*x_i^2 - K3*sum_{j<i}(s_j*x_j + s_j^3*x_j^3)
#                                         - K3*sum_{j>i}(s_j*x_j + s_j^2*x_j^2)
# Lower-right (I=N-UB..N): diagonal uses CB, all neighbors use SQ
#   v[I] = K1*s_i*x_i + K2*s_i^3*x_i^3 - K3*sum_j(s_j*x_j + s_j^2*x_j^2)
# Objective = sum_I v[I]^2
function ssbrybnd_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    K1 = T(2)
    K2 = T(5)
    K3 = T(1)
    LB = 5; UB = 1
    SCAL = T(6)

    RN1 = T(N - 1)
    S = T[exp(T(i - 1) / RN1 * SCAL) for i in 1:N]

    x0 = T[T(1) / S[i] for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # ── Upper left: I=1..LB ─────────────────────────────────────────────────
    # I=1: upper neighbors only = {2}
    ExaModels.@add_obj(c,
        (K1*S[1]*x[1] + K2*S[1]^3*x[1]^3
         - K3*(S[2]*x[2] + S[2]^2*x[2]^2))^2
    )
    # I=2: lower = {1}, upper = {3}
    ExaModels.@add_obj(c,
        (K1*S[2]*x[2] + K2*S[2]^3*x[2]^3
         - K3*(S[1]*x[1] + S[1]^2*x[1]^2)
         - K3*(S[3]*x[3] + S[3]^2*x[3]^2))^2
    )
    # I=3: lower = {1,2}, upper = {4}
    ExaModels.@add_obj(c,
        (K1*S[3]*x[3] + K2*S[3]^3*x[3]^3
         - K3*(S[1]*x[1] + S[1]^2*x[1]^2)
         - K3*(S[2]*x[2] + S[2]^2*x[2]^2)
         - K3*(S[4]*x[4] + S[4]^2*x[4]^2))^2
    )
    # I=4: lower = {1,2,3}, upper = {5}
    ExaModels.@add_obj(c,
        (K1*S[4]*x[4] + K2*S[4]^3*x[4]^3
         - K3*(S[1]*x[1] + S[1]^2*x[1]^2)
         - K3*(S[2]*x[2] + S[2]^2*x[2]^2)
         - K3*(S[3]*x[3] + S[3]^2*x[3]^2)
         - K3*(S[5]*x[5] + S[5]^2*x[5]^2))^2
    )
    # I=5: lower = {1,2,3,4}, upper = {6}
    ExaModels.@add_obj(c,
        (K1*S[5]*x[5] + K2*S[5]^3*x[5]^3
         - K3*(S[1]*x[1] + S[1]^2*x[1]^2)
         - K3*(S[2]*x[2] + S[2]^2*x[2]^2)
         - K3*(S[3]*x[3] + S[3]^2*x[3]^2)
         - K3*(S[4]*x[4] + S[4]^2*x[4]^2)
         - K3*(S[6]*x[6] + S[6]^2*x[6]^2))^2
    )

    # ── Main: I = LB+1..N-UB-1 = 6..N-2 ───────────────────────────────────
    # v[I] = K1*s_i*x_i + K2*s_i^2*x_i^2
    #       - K3*(s_{i-1}*x_{i-1} + s_{i-1}^3*x_{i-1}^3)
    #       - K3*(s_{i-2}*x_{i-2} + s_{i-2}^3*x_{i-2}^3)
    #       - K3*(s_{i-3}*x_{i-3} + s_{i-3}^3*x_{i-3}^3)
    #       - K3*(s_{i-4}*x_{i-4} + s_{i-4}^3*x_{i-4}^3)
    #       - K3*(s_{i-5}*x_{i-5} + s_{i-5}^3*x_{i-5}^3)
    #       - K3*(s_{i+1}*x_{i+1} + s_{i+1}^2*x_{i+1}^2)
    ExaModels.@add_par(c, Spar, S)
    ExaModels.@add_obj(c,
        (K1*Spar[i]*x[i] + K2*Spar[i]^2*x[i]^2
         - K3*(Spar[i-1]*x[i-1] + Spar[i-1]^3*x[i-1]^3)
         - K3*(Spar[i-2]*x[i-2] + Spar[i-2]^3*x[i-2]^3)
         - K3*(Spar[i-3]*x[i-3] + Spar[i-3]^3*x[i-3]^3)
         - K3*(Spar[i-4]*x[i-4] + Spar[i-4]^3*x[i-4]^3)
         - K3*(Spar[i-5]*x[i-5] + Spar[i-5]^3*x[i-5]^3)
         - K3*(Spar[i+1]*x[i+1] + Spar[i+1]^2*x[i+1]^2))^2
        for i in LB+1:N-UB-1
    )

    # ── Lower right: I = N-UB..N = N-1..N ──────────────────────────────────
    # I=N-1: lower = {N-6,N-5,N-4,N-3,N-2}, upper = {N}
    ExaModels.@add_obj(c,
        (K1*S[N-1]*x[N-1] + K2*S[N-1]^3*x[N-1]^3
         - K3*(S[N-6]*x[N-6] + S[N-6]^2*x[N-6]^2)
         - K3*(S[N-5]*x[N-5] + S[N-5]^2*x[N-5]^2)
         - K3*(S[N-4]*x[N-4] + S[N-4]^2*x[N-4]^2)
         - K3*(S[N-3]*x[N-3] + S[N-3]^2*x[N-3]^2)
         - K3*(S[N-2]*x[N-2] + S[N-2]^2*x[N-2]^2)
         - K3*(S[N]*x[N] + S[N]^2*x[N]^2))^2
    )
    # I=N: lower = {N-5,N-4,N-3,N-2,N-1}, upper = {}
    ExaModels.@add_obj(c,
        (K1*S[N]*x[N] + K2*S[N]^3*x[N]^3
         - K3*(S[N-5]*x[N-5] + S[N-5]^2*x[N-5]^2)
         - K3*(S[N-4]*x[N-4] + S[N-4]^2*x[N-4]^2)
         - K3*(S[N-3]*x[N-3] + S[N-3]^2*x[N-3]^2)
         - K3*(S[N-2]*x[N-2] + S[N-2]^2*x[N-2]^2)
         - K3*(S[N-1]*x[N-1] + S[N-1]^2*x[N-1]^2))^2
    )

    return ExaModels.ExaModel(c; kwargs...)
end
