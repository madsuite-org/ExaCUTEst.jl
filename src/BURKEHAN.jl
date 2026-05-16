# BURKEHAN: An infeasible QP (Burke & Han, 1989)
# minimize  x
# subject to  x^2 <= -1  (i.e. x^2 + 1 <= 0, L group with rhs -1)
# 1 variable X, free (XM bound = MI = lvar=-Inf), uvar=0
# CON: L group, elements SQ(x), rhs = -1  → x^2 - (-1) <= 0  → x^2 + 1 <= 0
# SQ element: F = V1*V1
function burkehan_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # XM bound: lvar=-Inf, XU bound: uvar=0
    ExaModels.@add_var(c, x, 1; lvar = T(-Inf), uvar = T(0), start = T(10))
    # Objective: x (linear group OBJ with X coeff 1.0)
    ExaModels.@add_obj(c, x[1])
    # Constraint CON (L type, ≤ 0): SQ(x) - (-1) ≤ 0 → x^2 + 1 ≤ 0
    # rhs = -1, so constraint value = x^2 - (-1) = x^2 + 1 ≤ 0
    ExaModels.@add_con(c, x[1]^2 - T(-1); lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
