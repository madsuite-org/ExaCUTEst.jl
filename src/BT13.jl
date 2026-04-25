function bt13_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # 5 variables: X1-X4 free, X5 >= 0
    # Start: X1=1, X2=2, X3=3, X4=3, X5=228
    ExaModels.@add_var(c, x, 5;
                       lvar = T[-Inf, -Inf, -Inf, -Inf, 0.0],
                       uvar = T[Inf, Inf, Inf, Inf, Inf],
                       start = T[1.0, 2.0, 3.0, 3.0, 228.0])
    # OBJ (N, identity): X5
    ExaModels.@add_obj(c, x[5])
    # CON1 (E): X1^2 + (X1-2*X2)^2 + (X2-3*X3)^2 + (X3-4*X4)^2 - X5^2 = 0
    ExaModels.@add_con(c, x[1]^2 + (x[1] - T(2)*x[2])^2 + (x[2] - T(3)*x[3])^2 + (x[3] - T(4)*x[4])^2 - x[5]^2)
    return ExaModels.ExaModel(c; kwargs...)
end
