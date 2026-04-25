# SBRYBND: Scaled Broyden banded system, least-squares sense (N=5000, all free)
# Scaled version of BRYBND: SCALE(I) = exp((I-1)/(N-1)*12)
# Start: X(I) = 1/SCALE(I) = exp(-(I-1)/(N-1)*12)
# KAPPA1=2, KAPPA2=5, KAPPA3=1, LB=5, UB=1
#
# Group G(I) value (L2 objective, contribution = v^2):
#   v = K1*S[I]*x[I]  (linear from GROUPS)
#     - K3 * sum_{neighbors j} S[J]*x[J]  (linear from GROUPS)
#     + K2*S[I]*x[I]^2 or K2*S[I]*x[I]^3  (SQ/CB diagonal element)
#     - K3 * sum_{lower neighbors j} S[J]*x[J]^3 or S[J]*x[J]^2  (off-diagonal elements)
#   Upper/lower regions use SQ (scale*V^2) for off-diagonal and CB (scale*V^3) for diagonal
#   Main region uses CB for lower band, SQ for diagonal and upper band
function sbrybnd_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    K1 = T(2); K2 = T(5); K3 = T(1)
    LB = 5; UB = 1
    SCAL = T(12)
    S = T[exp(T(i - 1) / T(N - 1) * SCAL) for i in 1:N]
    x0 = T[T(1) / S[i] for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)
    ExaModels.@add_par(c, SP, S)  # SCALE parameter array

    # Upper left: I=1..LB (use CB for diagonal, SQ for neighbors)
    # v = K1*S[I]*x[I] + K2*S[I]*x[I]^3  (linear + CB diagonal)
    #   - K3*sum_{j=1..I-1} S[J]*(x[J] + x[J]^2)  (linear + SQ for lower neighbors)
    #   - K3*sum_{j=I+1..I+UB} S[J]*(x[J] + x[J]^2)  (linear + SQ for upper neighbors)
    ExaModels.@add_obj(c,
        (K1*SP[1]*x[1] + K2*SP[1]*x[1]^3
         - K3*SP[2]*(x[2] + x[2]^2))^2)
    ExaModels.@add_obj(c,
        (K1*SP[2]*x[2] + K2*SP[2]*x[2]^3
         - K3*SP[1]*(x[1] + x[1]^2) - K3*SP[3]*(x[3] + x[3]^2))^2)
    ExaModels.@add_obj(c,
        (K1*SP[3]*x[3] + K2*SP[3]*x[3]^3
         - K3*SP[1]*(x[1]+x[1]^2) - K3*SP[2]*(x[2]+x[2]^2) - K3*SP[4]*(x[4]+x[4]^2))^2)
    ExaModels.@add_obj(c,
        (K1*SP[4]*x[4] + K2*SP[4]*x[4]^3
         - K3*SP[1]*(x[1]+x[1]^2) - K3*SP[2]*(x[2]+x[2]^2) - K3*SP[3]*(x[3]+x[3]^2)
         - K3*SP[5]*(x[5]+x[5]^2))^2)
    ExaModels.@add_obj(c,
        (K1*SP[5]*x[5] + K2*SP[5]*x[5]^3
         - K3*SP[1]*(x[1]+x[1]^2) - K3*SP[2]*(x[2]+x[2]^2) - K3*SP[3]*(x[3]+x[3]^2)
         - K3*SP[4]*(x[4]+x[4]^2) - K3*SP[6]*(x[6]+x[6]^2))^2)

    # Main: I=LB+1..N-UB-1 (use CB for lower band, SQ for diagonal and upper band)
    # v = K1*S[I]*x[I] + K2*S[I]*x[I]^2  (linear + SQ diagonal)
    #   - K3*sum_{j=I-LB..I-1} S[J]*(x[J] + x[J]^3)  (linear + CB for lower band)
    #   - K3*S[I+1]*(x[I+1] + x[I+1]^2)  (linear + SQ for upper neighbor)
    ExaModels.@add_obj(c,
        (K1*SP[i]*x[i] + K2*SP[i]*x[i]^2
         - K3*(SP[i-5]*(x[i-5]+x[i-5]^3) + SP[i-4]*(x[i-4]+x[i-4]^3)
             + SP[i-3]*(x[i-3]+x[i-3]^3) + SP[i-2]*(x[i-2]+x[i-2]^3)
             + SP[i-1]*(x[i-1]+x[i-1]^3))
         - K3*SP[i+1]*(x[i+1]+x[i+1]^2))^2
        for i in LB+1:N-UB-1)

    # Lower right: I=N-UB..N (use CB for diagonal, SQ for neighbors)
    # I=N-1: lower = {N-6..N-2}, upper = {N}
    ExaModels.@add_obj(c,
        (K1*SP[N-1]*x[N-1] + K2*SP[N-1]*x[N-1]^3
         - K3*(SP[N-6]*(x[N-6]+x[N-6]^2) + SP[N-5]*(x[N-5]+x[N-5]^2)
             + SP[N-4]*(x[N-4]+x[N-4]^2) + SP[N-3]*(x[N-3]+x[N-3]^2)
             + SP[N-2]*(x[N-2]+x[N-2]^2))
         - K3*SP[N]*(x[N]+x[N]^2))^2)
    # I=N: lower = {N-5..N-1}, no upper neighbors
    ExaModels.@add_obj(c,
        (K1*SP[N]*x[N] + K2*SP[N]*x[N]^3
         - K3*(SP[N-5]*(x[N-5]+x[N-5]^2) + SP[N-4]*(x[N-4]+x[N-4]^2)
             + SP[N-3]*(x[N-3]+x[N-3]^2) + SP[N-2]*(x[N-2]+x[N-2]^2)
             + SP[N-1]*(x[N-1]+x[N-1]^2)))^2)

    return ExaModels.ExaModel(c; kwargs...)
end
