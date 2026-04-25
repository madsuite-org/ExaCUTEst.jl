function bt2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # 3 variables, all free, start = 10.0
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf, -Inf, -Inf], uvar = T[Inf, Inf, Inf],
                       start = T[10.0, 10.0, 10.0])
    # OBJ (N, identity): SSQ(X1,P=1) + ISQ(X1,X2) + I4TH(X2,X3)
    #   SSQ(X1,P=1): (X1-1)^2
    #   ISQ(X1,X2) with IV Z=X1-X2: Z^2 = (X1-X2)^2
    #   I4TH(X2,X3) with IV Z=X2-X3: Z^4
    ExaModels.@add_obj(c, (x[1] - T(1))^2 + (x[1] - x[2])^2 + (x[2] - x[3])^4)
    # CON1 (E): MISC(X1,X2) + FOURTH(X3) - 8.2426407 = 0
    #   MISC(X1,X2): X1*(1+X2^2)
    #   FOURTH(X3): X3^4
    ExaModels.@add_con(c, x[1] * (T(1) + x[2]^2) + x[3]^4 - T(8.2426407))
    return ExaModels.ExaModel(c; kwargs...)
end
