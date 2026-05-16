function wachbieg_model(; T = Float64, backend = nothing, kwargs...)
    # WACHBIEG: 3-variable problem
    # Parameters: A=-1.0, B=0.5, -A=1.0
    # Variables: X1 (FR, free), X2 ≥ 0, X3 ≥ 0
    # Objective: X1
    # CON1 (E): X1^2 - X2 = -(-A) = 1.0  → X1^2 - X2 - 1 = 0
    #   (E CON1 has -X2, element E1=X1^2, RHS = -A = 1.0)
    #   v_CON1 = X1^2 - X2 - 1 = 0
    # CON2 (E): X1 - X3 = B = 0.5
    #   v_CON2 = X1 - X3 - 0.5 = 0

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[-Inf, 0, 0],
        uvar = T[Inf, Inf, Inf],
        start = T[-2.0, 1.0, 1.0])

    # Objective: X1
    ExaModels.@add_obj(c, x[1])

    # CON1 (E): X1^2 - X2 - 1 = 0
    ExaModels.@add_con(c, x[1]^2 - x[2] - T(1))

    # CON2 (E): X1 - X3 - 0.5 = 0
    ExaModels.@add_con(c, x[1] - x[3] - T(0.5))

    return ExaModels.ExaModel(c; kwargs...)
end
