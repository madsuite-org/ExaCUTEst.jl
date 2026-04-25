function himmelbh_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[0.0, 2.0])
    # Group G1 (identity, no GROUP TYPE):
    # linear: -3*X1 - 2*X2
    # elements: E1 = X1^3, E2 = X2^2
    # CONSTANT G1 = -2.0 → v = linear + elements - CONSTANT = -3*x1 - 2*x2 + x1^3 + x2^2 - (-2.0)
    # objective = v (identity group type)
    ExaModels.@add_obj(c, x[1]^3 + x[2]^2 - T(3) * x[1] - T(2) * x[2] + T(2))
    return ExaModels.ExaModel(c; kwargs...)
end
