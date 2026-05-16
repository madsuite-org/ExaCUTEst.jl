# OSCIGRAD: Roots of gradient of Nesterov oscillating path problem
# SUR2 (sum of squares unconstrained)
# Objective: sum of squared group values
# G1 group: v = 0.5*x1 - 0.5 (constant RHS)  → L2: (0.5*x1-0.5)^2
# G(I) for I=2..N: v = A(I) + B(I) where
#   A(I) = P*(X(I) - 2*X(I-1)^2 + 1)   (element A, P=2*RHO)
#   B(I) = P*(X(I+1) - 2*X(I)^2 + 1)*X(I) (element B for I=2..N-1)
# G(N): just A(N)
# All group types L2: contribution = v^2
function oscigrad_model(N = 100000; T = Float64, backend = nothing, kwargs...)
    RHO = T(500.0)
    TWO_RHO = T(2) * RHO
    NEG_FOUR_RHO = T(-4) * RHO

    start = fill(T(0.0), N)
    start[1] = T(-2.0)
    for i in 2:N
        start[i] = T(1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # G1: (0.5*x1 - 0.5)^2 (L2 of linear group)
    # The group value is 0.5*x[1] - rhs where rhs=0.5, so v=0.5*x[1]-0.5
    # L2 group type: contribution = v^2
    ExaModels.@add_obj(c, (T(0.5) * x[1] - T(0.5) + TWO_RHO * (x[2] - T(2) * x[1]^2 + T(1)) * x[1])^2)

    # G(I) for I=2..N-1: v = A(I) + B(I)
    # A(I): 2*RHO*(X(I) - 2*X(I-1)^2 + 1)
    # B(I): -4*RHO*(X(I+1) - 2*X(I)^2 + 1)*X(I)
    # SIF: element A with P=2RHO: F = P*(V-2U^2+1), G_V=P, G_U=-4PU
    # element B with P=-4RHO: F = P*(V-2U^2+1)*U
    ExaModels.@add_obj(c,
        (TWO_RHO * (x[i] - T(2) * x[i-1]^2 + T(1)) +
         NEG_FOUR_RHO * (x[i+1] - T(2) * x[i]^2 + T(1)) * x[i])^2
        for i in 2:N-1)

    # G(N): just A(N)
    ExaModels.@add_obj(c, (TWO_RHO * (x[N] - T(2) * x[N-1]^2 + T(1)))^2)

    return ExaModels.ExaModel(c; kwargs...)
end
