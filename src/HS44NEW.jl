# HS44NEW: HS44 variant with different element ordering (Hock & Schittkowski 1981)
# 4 variables: X1..X4 ≥ 0; start = 1.0
# Objective: X1 - X2 - X3 - X1*X3 + X1*X4 + X2*X3 - X2*X4
# 6 G-type (linear) constraints
function hs44new_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T(0), uvar = T(Inf), start = T(1))
    ExaModels.@add_obj(c,
        x[1] - x[2] - x[3]
        - x[1]*x[3] + x[1]*x[4] + x[2]*x[3] - x[2]*x[4])
    ExaModels.@add_con(c, -x[1] - T(2)*x[2] + T(8); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -T(4)*x[1] - x[2] + T(12); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -T(3)*x[1] - T(4)*x[2] + T(12); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -T(2)*x[3] - x[4] + T(8); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[3] - T(2)*x[4] + T(8); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[3] - x[4] + T(5); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
