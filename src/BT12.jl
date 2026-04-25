function bt12_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # 5 variables, all free
    # Start: XV 'DEFAULT' 2.0 then individual overrides
    # X1=15.811, X2=1.5811, X3=0.0, X4=15.083, X5=3.7164
    ExaModels.@add_var(c, x, 5; lvar = T(-Inf), uvar = T(Inf),
                       start = T[15.811, 1.5811, 0.0, 15.083, 3.7164])
    # OBJ: 0.01*X1^2 + X2^2
    ExaModels.@add_obj(c, T(0.01)*x[1]^2 + x[2]^2)
    # CON1 (E): X1 + X2 - X3^2 - 25 = 0
    ExaModels.@add_con(c, x[1] + x[2] - x[3]^2 - T(25))
    # CON2 (E): X1^2 + X2^2 - X4^2 - 25 = 0
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - x[4]^2 - T(25))
    # CON3 (E): X1 - X5^2 - 2 = 0
    ExaModels.@add_con(c, x[1] - x[5]^2 - T(2))
    return ExaModels.ExaModel(c; kwargs...)
end
