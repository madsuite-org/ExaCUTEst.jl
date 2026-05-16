function zy2_model(; T = Float64, backend = nothing, kwargs...)
    # ZY2: 3-variable polynomial program
    # Objective: X1^3 - 6*X1^2 + 11*X1 + X2 + X3
    # (OBJ linear: 11*X1 + X2 + X3, elements: X1CB=X1^3, X1SQ=-6*X1^2)
    # C1 (L, ≤ 10): X1^2 + X2^2 + X3^2 ≤ 10  → residual ≤ 0
    # C2 (G, ≥ 4) : X1^2 + X2^2 + X3^2 ≥ 4   → residual ≥ 0
    # Bounds: X1,X2 ≥ 0; X3 ∈ [0, 5]; X1,X2 free? default x≥0
    # SIF BOUNDS: only XU X3 5.0 → X1,X2 default lvar=0; X3 ∈ [0,5]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[0, 0, 0],
        uvar = T[Inf, Inf, 5],
        start = T[0.1, 0.1, 3.0])

    # Objective: X1^3 - 6*X1^2 + 11*X1 + X2 + X3
    ExaModels.@add_obj(c, x[1]^3 - T(6)*x[1]^2 + T(11)*x[1] + x[2] + x[3])

    # C1 (L ≤ 10): x1^2 + x2^2 + x3^2 - 10 ≤ 0
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 + x[3]^2 - T(10); lcon = T(-Inf), ucon = T(0))

    # C2 (G ≥ 4): x1^2 + x2^2 + x3^2 - 4 ≥ 0
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 + x[3]^2 - T(4); lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
