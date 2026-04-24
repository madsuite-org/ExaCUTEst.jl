# ROSENBR: Rosenbrock "banana valley" (2 variables, unconstrained)
# minimize 100*(x2 - x1^2)^2 + (x1 - 1)^2
# Source: More', Garbow, Hillstrom (1981), problem 1
# SIF: N group, L2 group type (phi=v^2), element SQ (v1^2) with coeff -1, scale 0.01
#   G1: sigma=0.01, L2(x2 - x1^2) → (x2-x1^2)^2 / 0.01 = 100*(x2-x1^2)^2
#   G2: L2(x1 - 1) → (x1-1)^2
function rosenbr_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; start = T[-1.2, 1.0])
    ExaModels.@add_obj(c, 100 * (x[2] - x[1]^2)^2 + (x[1] - 1)^2)
    return ExaModels.ExaModel(c; kwargs...)
end
