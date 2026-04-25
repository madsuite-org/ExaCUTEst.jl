# ZANGWIL2: Zangwill's problem in 2 variables (unconstrained)
# Source: Buckley (1989), problem 7, p.102
# SIF: single N group G1 with SCALE=15 and no GROUP TYPE (identity).
#   Linear: -56*x1 - 256*x2
#   Elements: 16*x1^2 + 16*x2^2 - 8*x1*x2
#   Constant RHS: -991  →  constant subtracted: +991
#   With SCALE=15 and identity group type: contribution = v / 15
#   Objective = (16*x1^2 + 16*x2^2 - 8*x1*x2 - 56*x1 - 256*x2 + 991) / 15
function zangwil2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[3.0, 8.0])
    ExaModels.@add_obj(
        c,
        (T(16) * x[1]^2 + T(16) * x[2]^2 - T(8) * x[1] * x[2]
         - T(56) * x[1] - T(256) * x[2] + T(991)) / T(15)
    )
    return ExaModels.ExaModel(c; kwargs...)
end
