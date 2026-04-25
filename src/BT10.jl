function bt10_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # 2 variables, all free, start = 2.0
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T(2.0))
    # OBJ (N, identity): -X1
    ExaModels.@add_obj(c, -x[1])
    # CON1 (E): X2 - X1^3 = 0
    ExaModels.@add_con(c, x[2] - x[1]^3)
    # CON2 (E): -X2 + X1^2 = 0
    ExaModels.@add_con(c, -x[2] + x[1]^2)
    return ExaModels.ExaModel(c; kwargs...)
end
