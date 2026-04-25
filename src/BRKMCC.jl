# BRKMCC: Buckley problem 85 (2 variables, unconstrained)
# Source: A.R. Buckley, TR 1989CS-3, Dalhousie University, 1989.
# SIF input: Ph. Toint, Dec 1989.
#
# SIF group analysis (contribution = F(v) / sigma):
#   G1: L2 type (no scale), linear X1, rhs=2  → v = X1-2, contrib = v^2 = (X1-2)^2
#   G2: L2 type (no scale), linear X2, rhs=1  → v = X2-1, contrib = v^2 = (X2-1)^2
#   G3: INV type, SCALE=25, elements E1=-0.25*X1^2, E2=-1.0*X2^2, rhs=-1
#       v = -0.25*X1^2 - X2^2 - (-1) = 1 - 0.25*X1^2 - X2^2
#       INV: F(v) = 1/v, with scale: F(v)/sigma = 1/(v*25)
#       contrib = 1 / (25 * (1 - 0.25*X1^2 - X2^2))
#   G4: L2 type, SCALE=0.2, linear X1 - 2*X2, rhs=-1
#       v = X1 - 2*X2 - (-1) = X1 - 2*X2 + 1
#       contrib = v^2 / 0.2 = 5*(X1-2*X2+1)^2
function brkmcc_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf], start = T[2.0, 2.0])
    ExaModels.@add_obj(c, (x[1] - T(2))^2)
    ExaModels.@add_obj(c, (x[2] - T(1))^2)
    ExaModels.@add_obj(c, T(1) / (T(25) * (T(1) - T(0.25) * x[1]^2 - x[2]^2)))
    ExaModels.@add_obj(c, T(5) * (x[1] - T(2) * x[2] + T(1))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
