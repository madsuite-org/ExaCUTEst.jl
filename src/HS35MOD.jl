# HS35MOD: HS35 with X2 fixed at 0.5 (Hock & Schittkowski 1981)
# 3 variables: X1,X3 ∈ [0,∞), X2 = 0.5 (fixed); start = 0.5
# Objective: 2X1² + 2X2² + X3² + 2X1X2 + 2X1X3 - 8X1 - 6X2 - 4X3 + 9
# Constraint: -X1 - X2 - 2X3 + 3 ≥ 0
function hs35mod_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[T(0), T(0.5), T(0)],
        uvar = T[T(Inf), T(0.5), T(Inf)],
        start = T(0.5))
    ExaModels.@add_obj(c,
        T(2)*x[1]^2 + T(2)*x[2]^2 + x[3]^2 + T(2)*x[1]*x[2] + T(2)*x[1]*x[3]
        - T(8)*x[1] - T(6)*x[2] - T(4)*x[3] + T(9))
    ExaModels.@add_con(c, -x[1] - x[2] - T(2)*x[3] + T(3); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
