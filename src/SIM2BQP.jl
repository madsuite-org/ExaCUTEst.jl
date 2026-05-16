# SIM2BQP: Another simple box-constrained quadratic (2 variables)
# Fixed: x1=0 (FX 'DEFAULT' means all fixed, then LO/UP override x2)
# Actually FX SIM2BQP 'DEFAULT' fixes all to 0; then x2 gets lvar=0, uvar=0.5
# Wait: FX sets lvar=uvar=0 for all, then LO x2 0.0 and UP x2 0.5 sets x2 bounds.
# But x1 is fixed at 0.
# min  x2 + (x2-x1)^2 + (2*x1+x2)^2
# x1 fixed at 0, x2 in [0, 0.5]
function sim2bqp_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # x1 is fixed at 0 (FX 'DEFAULT' then only x2 gets different bounds)
    ExaModels.@add_var(c, x, 2;
                       lvar = T[0.0, 0.0], uvar = T[0.0, 0.5],
                       start = T[10.0, 1.0])
    # G1: identity group, v = x2
    ExaModels.@add_obj(c, x[2])
    # G2: L2 group, v = -x1 + x2 => v^2 = (x2 - x1)^2
    ExaModels.@add_obj(c, (x[2] - x[1])^2)
    # G3: L2 group, v = 2*x1 + x2 => v^2 = (2*x1 + x2)^2
    ExaModels.@add_obj(c, (T(2) * x[1] + x[2])^2)
    return ExaModels.ExaModel(c; kwargs...)
end
