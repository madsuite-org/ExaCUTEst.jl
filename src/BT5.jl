function bt5_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # 3 variables, all free, start = 2.0
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf, -Inf, -Inf], uvar = T[Inf, Inf, Inf],
                       start = T[2.0, 2.0, 2.0])
    # OBJ (N, identity): elements with coefficients + constant
    #   Elements: X1SQ(-1) + X3SQ(-1) + X2SQ(-2) + X1X2(-1) + X1X3(-1)
    #   Constant (RHS): -1000 → group_value = element_sum - rhs
    #   group_value = -X1^2 - X3^2 - 2*X2^2 - X1*X2 - X1*X3 - (-1000)
    #               = -X1^2 - 2*X2^2 - X3^2 - X1*X2 - X1*X3 + 1000
    ExaModels.@add_obj(c, -x[1]^2 - T(2)*x[2]^2 - x[3]^2 - x[1]*x[2] - x[1]*x[3] + T(1000))
    # CON1 (E): X1^2 + X2^2 + X3^2 - 25 = 0
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 + x[3]^2 - T(25))
    # CON2 (E): 8*X1 + 14*X2 + 7*X3 - 56 = 0
    ExaModels.@add_con(c, T(8)*x[1] + T(14)*x[2] + T(7)*x[3] - T(56))
    return ExaModels.ExaModel(c; kwargs...)
end
