function bt4_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # 3 variables, all free
    # CUTEst variable ordering: x[1]=X2, x[2]=X1, x[3]=X3
    # Start: X1=4.0382, X2=-2.9470, X3=-0.09115
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf, -Inf, -Inf], uvar = T[Inf, Inf, Inf],
                       start = T[-2.9470, 4.0382, -0.09115])
    # OBJ (N, identity): linear X1-X2 + CB(X2)
    #   OBJ = X1 - X2 + X2^3 = x[2] - x[1] + x[1]^3
    ExaModels.@add_obj(c, x[2] - x[1] + x[1]^3)
    # CUTEst constraint ordering: linear-first (CON2 before CON1)
    # CON2 (E, linear): X1+X2+X3-1 = x[2]+x[1]+x[3]-1
    ExaModels.@add_con(c, x[2] + x[1] + x[3] - T(1))
    # CON1 (E, quadratic): X1^2+X2^2+X3^2-25 = x[2]^2+x[1]^2+x[3]^2-25
    ExaModels.@add_con(c, x[2]^2 + x[1]^2 + x[3]^2 - T(25))
    return ExaModels.ExaModel(c; kwargs...)
end
