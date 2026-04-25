# MODBEALE: Modified Beale problem (2*N2 variables, unconstrained)
# Default N2 = N/2 = 10000
# Source: Toint adaptation of More', Garbow, Hillstrom (1981), problem 5
# SIF:
#   PRODB element: F(V1,V2;POW) = V1*(1 - V2^POW)
#   BA(I): v = X(2I-1)*(1-X(2I)^1) - 1.5;   contribution = v^2
#   BB(I): v = X(2I-1)*(1-X(2I)^2) - 2.25;  contribution = v^2
#   BC(I): v = X(2I-1)*(1-X(2I)^3) - 2.625; contribution = v^2
#   L(I) for I=1..N/2-1: linear 6*X(2I) - X(2I+1), SCALE=1/50
#     contribution = (6*X(2I) - X(2I+1))^2 / (1/50) = 50*(6*X(2I)-X(2I+1))^2
#   BOUNDS: FR DEFAULT (free variables); all start at 1.0
function modbeale_model(N2 = 10000; T = Float64, backend = nothing, kwargs...)
    N = 2 * N2
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(T(-Inf), N), uvar = fill(T(Inf), N),
                       start = fill(T(1.0), N))
    # BA(I), BB(I), BC(I) groups for I = 1..N2
    ExaModels.@add_obj(c, (x[2*i-1] * (T(1) - x[2*i]) - T(1.5))^2 for i in 1:N2)
    ExaModels.@add_obj(c, (x[2*i-1] * (T(1) - x[2*i]^2) - T(2.25))^2 for i in 1:N2)
    ExaModels.@add_obj(c, (x[2*i-1] * (T(1) - x[2*i]^3) - T(2.625))^2 for i in 1:N2)
    # L(I) groups for I = 1..N2-1: SCALE=ALPHINV=1/50
    #   contribution = (6*x[2i] - x[2i+1])^2 * 50
    ExaModels.@add_obj(c, T(50) * (T(6) * x[2*i] - x[2*i+1])^2 for i in 1:N2-1)
    return ExaModels.ExaModel(c; kwargs...)
end
