function bt6_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # 5 variables, all free, start = 2.0
    ExaModels.@add_var(c, x, 5; lvar = fill(T(-Inf), 5), uvar = fill(T(Inf), 5),
                       start = fill(T(2.0), 5))
    # OBJ (N, identity):
    #   O1 = SSQ(X1,P=2): (X1-1)^2
    #   O2 = ISQ(X1,X2) Z=X1-X2: (X1-X2)^2
    #   O3 = SSQ(X3,P=2): (X3-1)^2
    #   O4 = SSQ(X4,P=4): (X4-1)^4
    #   O5 = SSQ(X5,P=6): (X5-1)^6
    ExaModels.@add_obj(c, (x[1] - T(1))^2 + (x[1] - x[2])^2 + (x[3] - T(1))^2 +
                          (x[4] - T(1))^4 + (x[5] - T(1))^6)
    # Constants (computed from SIF arithmetic):
    # ROOT2 = sqrt(2), 2ROOT2 = 2*sqrt(2), 8+ROOT2 = 8+sqrt(2)
    # CON1 (E): C1E1 + C1E2 - 2*sqrt(2) = 0
    #   C1E1 = XY2(X4,X1): X4*X1^2
    #   C1E2 = ISIN(X4,X5) Z=X4-X5: sin(X4-X5)
    ExaModels.@add_con(c, x[4] * x[1]^2 + sin(x[4] - x[5]) - T(2) * sqrt(T(2)))
    # CON2 (E): X2 + C2E1 - (8+sqrt(2)) = 0
    #   C2E1 = X4Y2(X3,X2): X3^4*X2^2
    ExaModels.@add_con(c, x[2] + x[3]^4 * x[2]^2 - (T(8) + sqrt(T(2))))
    return ExaModels.ExaModel(c; kwargs...)
end
