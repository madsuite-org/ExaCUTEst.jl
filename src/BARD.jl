function bard_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))

    # 3 variables, all free (FR BARD 'DEFAULT'), starting at 1.0
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf, -Inf, -Inf], uvar = T[Inf, Inf, Inf],
                       start = T[1.0, 1.0, 1.0])

    # Observed Y values (RHS constants for the 15 groups)
    ExaModels.@add_par(c, Y, T[0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35,
                                0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39])

    # Element parameters U[i], V[i], W[i] for i=1..15
    # For i=1..8:  U=i, V=16-i, W=i
    # For i=9..15: U=i, V=16-i, W=16-i
    ExaModels.@add_par(c, U, T[float(i)       for i in 1:15])
    ExaModels.@add_par(c, V, T[float(16 - i)  for i in 1:15])
    ExaModels.@add_par(c, W, T[i <= 8 ? float(i) : float(16 - i) for i in 1:15])

    # Group I: v_I = X1 + E(I) - Y[I]
    # where E(I) = U[I] / (V[I]*X2 + W[I]*X3)   (element BD with F = U/Z, Z = V*V1 + W*V2)
    # L2 GROUP TYPE (no scale): contribution = v_I^2
    ExaModels.@add_obj(c, (x[1] + U[i] / (V[i] * x[2] + W[i] * x[3]) - Y[i])^2 for i in 1:15)

    return ExaModels.ExaModel(c; kwargs...)
end
