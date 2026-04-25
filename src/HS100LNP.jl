# HS100LNP: Modified HS100 with inactive nonlinear constraints removed (Toint/Plantenga)
# 7 free variables; start = [1,2,0,4,0,1,1]; 2 equality constraints
# Objective: (X1-10)² + (X2-12)²/0.2 + (X4-11)²/0.333... + O5_identity
# O5: -10X6 - 8X7 + 10X5⁶ + 7X6²+X7⁴-4X6X7 + X3⁴
function hs100lnp_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 7; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(1), T(2), T(0), T(4), T(0), T(1), T(1)])
    ExaModels.@add_obj(c, (x[1] - T(10))^2)
    ExaModels.@add_obj(c, (x[2] - T(12))^2 / T(0.2))
    ExaModels.@add_obj(c, (x[4] - T(11))^2 / T(0.33333333333))
    ExaModels.@add_obj(c,
        - T(10)*x[6] - T(8)*x[7]
        + T(10)*x[5]^6 + T(7)*x[6]^2 + x[7]^4 - T(4)*x[6]*x[7] + x[3]^4)
    # C1: -X3 - 5X5 - 2X1² - 3X2⁴ - 4X4² + 127 = 0
    ExaModels.@add_con(c, -x[3] - T(5)*x[5] - T(2)*x[1]^2 - T(3)*x[2]^4 - T(4)*x[4]^2 + T(127))
    # C4: -5X6 + 11X7 - 4X1² - X2² + 3X1X2 - 2X3² = 0
    ExaModels.@add_con(c, -T(5)*x[6] + T(11)*x[7] - T(4)*x[1]^2 - x[2]^2 + T(3)*x[1]*x[2] - T(2)*x[3]^2)
    return ExaModels.ExaModel(c; kwargs...)
end
