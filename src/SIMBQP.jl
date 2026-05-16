# SIMBQP: Simple box-constrained quadratic (2 variables)
# Variables are free except x2 in [0, 0.5]
# min  x2 + (x2-x1)^2 + (2*x1+x2)^2
# Groups:
#   G1 (identity): x2
#   G2 (L2): x2 - x1
#   G3 (L2): 2*x1 + x2
function simbqp_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2;
                       lvar = T[-Inf, 0.0], uvar = T[Inf, 0.5],
                       start = T[10.0, 1.0])
    # G1: identity group, v = x2
    ExaModels.@add_obj(c, x[2])
    # G2: L2 group, v = -x1 + x2 => v^2 = (x2 - x1)^2
    ExaModels.@add_obj(c, (x[2] - x[1])^2)
    # G3: L2 group, v = 2*x1 + x2 => v^2 = (2*x1 + x2)^2
    ExaModels.@add_obj(c, (T(2) * x[1] + x[2])^2)
    return ExaModels.ExaModel(c; kwargs...)
end
