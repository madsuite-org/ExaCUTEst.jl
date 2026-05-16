# SSCOSINE: Scaled cosine function (N=10 default)
# Source: Luksan et al. scaled variant of COSINE by Gould
# Variables X(i) with scale S(i) = exp((i-1)/(N-1) * SCAL), SCAL=6
# Objective: sum_{i=1}^{N-1} cos(S(i)*X(i) - 0.5*S(i+1)*X(i+1))
#   where G(i) is a COS group type applied to linear value v(i) = -0.5*S(i+1)*X(i+1)
#   and element E(i) = SQ type with P=S(i): P^2*X(i)^2
#   group G(i) with element E(i): v(i) + S(i)^2*X(i)^2
#   Wait, the linear part is -0.5*S(i+1)*X(i+1) and element E(i)=SQ(V=X(i),P=S(i)) -> P^2*V^2 = S(i)^2*X(i)^2
#   Total group value v(i) = S(i)^2*X(i)^2 + (-0.5*S(i+1))*X(i+1) [no rhs]
#   Applied COS group type: cos(v(i))
# Hmm wait: SQ element F = PP * V * V = P^2 * X^2 (from ELEMENTS section: A PP = P*P; F = PP*V*V)
# So element E(i): F = S(i)^2 * X(i)^2
# Linear part of G(i): MULT = -0.5*S(i+1), ZN G(i) X(i+1) MULT
# So G(i) group value: v(i) = S(i)^2*X(i)^2 + (-0.5*S(i+1))*X(i+1)
# COS group type: contribution = cos(v(i))
# Bounds: all free; Start: X(i) = 1/S(i)
function sscosine_model(N = 10; T = Float64, backend = nothing, kwargs...)
    SCAL = T(6.0)
    # S(i) = exp((i-1)/(N-1) * SCAL)
    S = T[exp((i - 1) / (N - 1) * SCAL) for i in 1:N]
    start = T[1 / S[i] for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: sum_{i=1}^{N-1} cos(S(i)^2 * X(i)^2 + (-0.5*S(i+1)) * X(i+1))
    ExaModels.@add_par(c, Si,   T[S[i] for i in 1:N])
    ExaModels.@add_par(c, Si1,  T[S[i + 1] for i in 1:N-1])
    ExaModels.@add_obj(c,
        cos(Si[i]^2 * x[i]^2 + (T(-0.5) * Si1[i]) * x[i + 1])
        for i in 1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
