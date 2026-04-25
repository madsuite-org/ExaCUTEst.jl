function bt8_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # 5 variables, all free
    # Start: X1=X2=X3=1, X4=X5=0
    ExaModels.@add_var(c, x, 5; lvar = fill(T(-Inf), 5), uvar = fill(T(Inf), 5),
                       start = T[1.0, 1.0, 1.0, 0.0, 0.0])
    # OBJ (N, identity): X1^2 + X2^2 + X3^2
    ExaModels.@add_obj(c, x[1]^2 + x[2]^2 + x[3]^2)
    # CON1 (E): X1 + X4SQ(-1) + X2SQ - 1 = 0
    #   = X1 - X4^2 + X2^2 - 1 = 0
    ExaModels.@add_con(c, x[1] - x[4]^2 + x[2]^2 - T(1))
    # CON2 (E): X1SQ + X2SQ + X5SQ(-1) - 1 = 0
    #   = X1^2 + X2^2 - X5^2 - 1 = 0
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - x[5]^2 - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
