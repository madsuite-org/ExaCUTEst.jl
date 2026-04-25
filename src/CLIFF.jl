# CLIFF: "Cliff problem" (2 variables, unconstrained)
# minimize (0.01*x1 - 0.03)^2 + (-x1 + x2) + exp(20*(x1 - x2))
#
# SIF groups:
#   G1: L2 type, v = 0.01*x1 - 0.03 (constant 0.03 in CONSTANTS section)
#       contribution = (0.01*x1 - 0.03)^2
#   G2: identity group (no GROUP TYPE), v = -x1 + x2
#       contribution = -x1 + x2
#   G3: 20EXP group type F(v) = exp(20*v), v = x1 - x2
#       contribution = exp(20*(x1 - x2))
#
# Bounds: all free (FR CLIFF 'DEFAULT')
# Start: x1=0, x2=-1
# Source: Buckley (1989), problem 206
function cliff_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T[0.0, -1.0])
    # G1: L2 group, v = 0.01*x[1] - 0.03 → contribution = v^2
    ExaModels.@add_obj(c, (T(0.01) * x[1] - T(0.03))^2)
    # G2: identity group, v = -x[1] + x[2] → contribution = v
    ExaModels.@add_obj(c, -x[1] + x[2])
    # G3: 20EXP group type, F(v) = exp(20*v), v = x[1] - x[2]
    ExaModels.@add_obj(c, exp(T(20) * (x[1] - x[2])))
    return ExaModels.ExaModel(c; kwargs...)
end
