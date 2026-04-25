# BRKMCC: Buckley problem 85 (2 variables, unconstrained)
# Objective = (X1-2)^2 + (X2-1)^2 + 625/(1 - 0.25*X1^2 - X2^2) + 5*(X1 - 2*X2 + 1)^2
#
# SIF group analysis:
#   G1: L2 group, linear X1, rhs=2  → v = X1-2, contribution = (X1-2)^2
#   G2: L2 group, linear X2, rhs=1  → v = X2-1, contribution = (X2-1)^2
#   G3: INV group, SCALE=25, elements E1=-0.25*X1^2, E2=-1.0*X2^2, rhs=-1
#       v = -0.25*X1^2 - X2^2 - (-1) = 1 - 0.25*X1^2 - X2^2
#       contribution = sigma * F(v/sigma) = 25 * (1/(v/25)) = 25 * 25/v = 625/v
#   G4: L2 group, SCALE=0.2, linear X1 - 2*X2, rhs=-1
#       v = X1 - 2*X2 - (-1) = X1 - 2*X2 + 1
#       contribution = v^2 / sigma = (X1-2*X2+1)^2 / 0.2 = 5*(X1-2*X2+1)^2
function brkmcc_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf], start = T[2.0, 2.0])
    ExaModels.@add_obj(c, (x[1] - T(2))^2)
    ExaModels.@add_obj(c, (x[2] - T(1))^2)
    ExaModels.@add_obj(c, T(625) / (T(1) - T(0.25) * x[1]^2 - x[2]^2))
    ExaModels.@add_obj(c, T(5) * (x[1] - T(2) * x[2] + T(1))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
