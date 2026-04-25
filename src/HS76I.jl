# HS76I: HS76 with integer upper bounds (Hock & Schittkowski 1981)
# 4 variables: X1..X4 ∈ [0,5]; start = 0.5
# Objective: -X1 - 3X2 + X3 - X4 + X1² + 0.5X2² + X3² + 0.5X4² - X1X3 + X3X4
# 2 L-type constraints, 1 G-type constraint (all linear)
function hs76i_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T(0), uvar = T(5), start = T(0.5))
    ExaModels.@add_obj(c,
        -x[1] - T(3)*x[2] + x[3] - x[4]
        + x[1]^2 + T(0.5)*x[2]^2 + x[3]^2 + T(0.5)*x[4]^2
        - x[1]*x[3] + x[3]*x[4])
    ExaModels.@add_con(c, x[1] + T(2)*x[2] + x[3] + x[4] - T(5); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, T(3)*x[1] + x[2] + T(2)*x[3] - x[4] - T(4); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, x[2] + T(4)*x[3] - T(1.5); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
