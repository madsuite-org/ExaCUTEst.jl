# HS100MOD: Modified HS100 with G-type inequality constraints (Toint 1991)
# 7 free variables; start = [1,2,0,4,0,1,1]; 4 G-type constraints
# Objective: same as HS100LNP
function hs100mod_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 7; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(1), T(2), T(0), T(4), T(0), T(1), T(1)])
    ExaModels.@add_obj(c, (x[1] - T(10))^2)
    ExaModels.@add_obj(c, (x[2] - T(12))^2 / T(0.2))
    ExaModels.@add_obj(c, (x[4] - T(11))^2 / T(0.33333333333))
    ExaModels.@add_obj(c,
        - T(10)*x[6] - T(8)*x[7]
        + T(10)*x[5]^6 + T(7)*x[6]^2 + x[7]^4 - T(4)*x[6]*x[7] + x[3]^4)
    # C1 (G, rhs=-127): -X3 - 5X5 - 2X1² - 3X2⁴ - 4X4² + 127 ≥ 0
    ExaModels.@add_con(c, -x[3] - T(5)*x[5] - T(2)*x[1]^2 - T(3)*x[2]^4 - T(4)*x[4]^2 + T(127); lcon = T(0), ucon = T(Inf))
    # C2 (G, rhs=-282): -7X1 - 3X2 - X4 + X5 - 10X3² + 282 ≥ 0
    ExaModels.@add_con(c, -T(7)*x[1] - T(3)*x[2] - x[4] + x[5] - T(10)*x[3]^2 + T(282); lcon = T(0), ucon = T(Inf))
    # C3 (G, rhs=-196): -23X1 + 8X7 - X2² - 6X6² + 196 ≥ 0
    ExaModels.@add_con(c, -T(23)*x[1] + T(8)*x[7] - x[2]^2 - T(6)*x[6]^2 + T(196); lcon = T(0), ucon = T(Inf))
    # C4 (G, rhs=0): 11X7 + 587X4 + 391X5 + 2193X6 - 4X1² - X2² + 3X1X2 - 2X3² ≥ 0
    ExaModels.@add_con(c, T(11)*x[7] + T(587)*x[4] + T(391)*x[5] + T(2193)*x[6] - T(4)*x[1]^2 - x[2]^2 + T(3)*x[1]*x[2] - T(2)*x[3]^2; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
