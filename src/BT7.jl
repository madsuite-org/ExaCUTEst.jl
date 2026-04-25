function bt7_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # 5 variables, all free
    # Start: X1=-2, X2=1, X3=X4=X5=1
    ExaModels.@add_var(c, x, 5; lvar = fill(T(-Inf), 5), uvar = fill(T(Inf), 5),
                       start = T[-2.0, 1.0, 1.0, 1.0, 1.0])
    # OBJ1 (N, GROUP TYPE L2, SCALE=0.01):
    #   Linear: X2*1.0, Element: X1SQ(-1.0)
    #   Group value v = X2 - X1^2
    #   L2 with σ=0.01: contribution = v^2/σ = (X2-X1^2)^2/0.01 = 100*(X2-X1^2)^2
    ExaModels.@add_obj(c, T(100) * (x[2] - x[1]^2)^2)
    # OBJ2 (N, GROUP TYPE L2, no scale):
    #   Linear: X1*1.0, Constant(RHS)=1.0
    #   Group value v = X1 - 1
    #   L2: v^2 = (X1-1)^2
    ExaModels.@add_obj(c, (x[1] - T(1))^2)
    # CON1 (E): elements X3SQ(-1) + X1X2 - 1 = 0
    #   = -X3^2 + X1*X2 - 1 = 0
    ExaModels.@add_con(c, -x[3]^2 + x[1] * x[2] - T(1))
    # CON2 (E): linear X1 + elements X4SQ(-1) + X2SQ - 0 = 0
    #   = X1 - X4^2 + X2^2 = 0
    ExaModels.@add_con(c, x[1] - x[4]^2 + x[2]^2)
    # CON3 (E): linear X1 + elements X5SQ - 0.5 = 0
    #   = X1 + X5^2 - 0.5 = 0
    ExaModels.@add_con(c, x[1] + x[5]^2 - T(0.5))
    return ExaModels.ExaModel(c; kwargs...)
end
