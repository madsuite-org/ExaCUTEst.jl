function womflet_model(; T = Float64, backend = nothing, kwargs...)
    # WOMFLET: 3-variable minmax problem
    # Variables: X1, X2, U (all free: FR DEFAULT)
    # Start: X1=3, X2=1, U=7.5
    # Objective: U
    # Elements: SQ(X2)=X2^2, RAT(X1)=X1/(X1+0.1)
    # C1 (G≥0): U - 0.5*X1 - X2^2 - 5*(X1/(X1+0.1)) ≥ 0
    # C2 (G≥0): U + 0.5*X1 - X2^2 - 5*(X1/(X1+0.1)) ≥ 0
    # C3 (G≥0): U + 0.5*X1 + X2^2 + 5*(X1/(X1+0.1)) ≥ 0
    # (indices: X1=1, X2=2, U=3)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[-Inf, -Inf, -Inf],
        uvar = T[Inf, Inf, Inf],
        start = T[3.0, 1.0, 7.5])

    # Objective: U (= x[3])
    ExaModels.@add_obj(c, x[3])

    # C1 (G≥0): U - 0.5*X1 - X2^2 - 5*(X1/(X1+0.1)) ≥ 0
    ExaModels.@add_con(c,
        x[3] - T(0.5)*x[1] - x[2]^2 - T(5)*x[1]/(x[1] + T(0.1));
        lcon = T(0), ucon = T(Inf))

    # C2 (G≥0): U + 0.5*X1 - X2^2 - 5*(X1/(X1+0.1)) ≥ 0
    ExaModels.@add_con(c,
        x[3] + T(0.5)*x[1] - x[2]^2 - T(5)*x[1]/(x[1] + T(0.1));
        lcon = T(0), ucon = T(Inf))

    # C3 (G≥0): U + 0.5*X1 + X2^2 + 5*(X1/(X1+0.1)) ≥ 0
    ExaModels.@add_con(c,
        x[3] + T(0.5)*x[1] + x[2]^2 + T(5)*x[1]/(x[1] + T(0.1));
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
