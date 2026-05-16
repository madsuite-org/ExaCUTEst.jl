# FLT: Filter method test problem, 2 variables, 2 constraints
# Minimize (X2 - 1)^2 subject to X1^2 = 0, X1^3 = 0
# Variables: X1, X2, both free; Start: X1=1, X2=0
function flt_model(; T = Float64, backend = nothing, kwargs...)
    NV = 2
    x0 = T[1, 0]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # Objective: OBJ group is L2 type with linear x2 - 1 → (x2 - 1)^2
    ExaModels.@add_obj(c, (x[2] - T(1))^2)

    # CON1 (E): x1^2 = 0
    ExaModels.@add_con(c, x[1]^2)

    # CON2 (E): x1^3 = 0
    ExaModels.@add_con(c, x[1]^3)

    return ExaModels.ExaModel(c; kwargs...)
end
