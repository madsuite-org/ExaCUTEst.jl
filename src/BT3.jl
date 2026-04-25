function bt3_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # 5 variables, all free, start = 20.0
    ExaModels.@add_var(c, x, 5; lvar = fill(T(-Inf), 5), uvar = fill(T(Inf), 5),
                       start = fill(T(20.0), 5))
    # OBJ (N, identity): ISQ(X1,X2) + SISQ(X2,X3,P=2) + SSQ(X4,P=1) + SSQ(X5,P=1)
    #   ISQ(X1,X2) with IV Z=X1-X2: Z^2 = (X1-X2)^2
    #   SISQ(X2,X3,P=2) with IV Z=X2+X3, T=Z-P=(X2+X3-2): T^2 = (X2+X3-2)^2
    #   SSQ(X4,P=1): (X4-1)^2
    #   SSQ(X5,P=1): (X5-1)^2
    ExaModels.@add_obj(c, (x[1] - x[2])^2 + (x[2] + x[3] - T(2))^2 +
                          (x[4] - T(1))^2 + (x[5] - T(1))^2)
    # CON1 (E): X1 + 3*X2 = 0  (no elements, no constant)
    ExaModels.@add_con(c, x[1] + T(3) * x[2])
    # CON2 (E): X3 + X4 - 2*X5 = 0
    ExaModels.@add_con(c, x[3] + x[4] - T(2) * x[5])
    # CON3 (E): X2 - X5 = 0
    ExaModels.@add_con(c, x[2] - x[5])
    return ExaModels.ExaModel(c; kwargs...)
end
