# TWOBARS: Two-bar structure analysis (Zhang & Fleury 1994)
# 2 variables: X1 ∈ [0.2, 4.0] (bar weight), X2 ∈ [0.1, 1.6] (spread); start = 1.0
# Objective: X1 * sqrt(1 + X2²)
# CE1 = sqrt(1+X2²) * (8/X1 + 1/(X1*X2))
# CE2 = sqrt(1+X2²) * (8/X1 - 1/(X1*X2))
# CONS1: 0.124 * CE1 - 1 ≤ 0
# CONS2: 0.124 * CE2 - 1 ≤ 0
function twobars_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[T(0.2), T(0.1)], uvar = T[T(4.0), T(1.6)],
        start = T(1))
    ExaModels.@add_obj(c, x[1] * sqrt(T(1) + x[2]^2))
    ExaModels.@add_con(c,
        T(0.124) * sqrt(T(1) + x[2]^2) * (T(8)/x[1] + T(1)/(x[1]*x[2])) - T(1);
        lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c,
        T(0.124) * sqrt(T(1) + x[2]^2) * (T(8)/x[1] - T(1)/(x[1]*x[2])) - T(1);
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
