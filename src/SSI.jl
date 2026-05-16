# SSI: Contrived infeasible pair of cubic equations (3 variables, unconstrained)
# Source: Nick Gould, Nov 2016; classification SUR2-AN-3-0
# GROUP TYPE L2: f(v) = 0.5*v^2
# C1: v = x1^2*x3 - 4; contribution = 0.5*(x1^2*x3 - 4)^2
# C2: v = x2^2 + x3;   contribution = 0.5*(x2^2 + x3)^2
# All free; start: x = 1.0
function ssi_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf), start = T(1))
    # C1: L2 group, v = x1^2*x3 - 4
    ExaModels.@add_obj(c, T(0.5) * (x[1]^2 * x[3] - T(4))^2)
    # C2: L2 group, v = x2^2 + x3
    ExaModels.@add_obj(c, T(0.5) * (x[2]^2 + x[3])^2)
    return ExaModels.ExaModel(c; kwargs...)
end
