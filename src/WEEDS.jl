function weeds_model(; T = Float64, backend = nothing, kwargs...)
    # WEEDS: Logistic growth model fit, 3 vars, 12 residuals
    # Model: B1 / (1 + B2*exp(-B3*TIME)) ≈ Y(I)
    # Objective: sum_{I=1}^12 (B1/(1+B2*exp(-B3*I)) - Y(I))^2
    # Variables: B1≥0, B2≥0, B3∈[0,3] (default lvar=0, uvar for B3=3)
    # Start: all 1.0

    Y_data = T[5.308, 7.24, 9.638, 12.866, 17.069, 23.192,
               31.443, 38.558, 50.156, 62.948, 75.995, 91.972]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[0, 0, 0],
        uvar = T[Inf, Inf, 3.0],
        start = T[1.0, 1.0, 1.0])

    # x[1]=B1, x[2]=B2, x[3]=B3
    ExaModels.@add_par(c, Y_par, Y_data)
    ExaModels.@add_par(c, T_par, T[T(i) for i in 1:12])
    ExaModels.@add_obj(c,
        (x[1] / (T(1) + x[2] * exp(-x[3] * T_par[i])) - Y_par[i])^2
        for i in 1:12)

    return ExaModels.ExaModel(c; kwargs...)
end
