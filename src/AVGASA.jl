# AVGASA: LP/QP avgas problem with integer variables treated as continuous (Leyffer 1998)
# 8 variables X1..X8 ∈ [0,1], start = 0.5
# Objective: quadratic with linear terms
# 10 G-type constraints (all linear, no elements)
function avgasa_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 8; lvar = T(0), uvar = T(1), start = T(0.5))
    ExaModels.@add_obj(c,
        - T(2)*x[2] - x[3] - T(3)*x[4] - T(2)*x[5] - T(4)*x[6] - T(3)*x[7] - T(5)*x[8]
        + T(2)*x[1]^2 + T(2)*x[2]^2 + T(2.5)*x[3]^2 + T(1.5)*x[4]^2
        + T(2.5)*x[5]^2 + T(2)*x[6]^2 + T(2.5)*x[7]^2 + T(2.5)*x[8]^2
        - x[1]*x[2] - T(2)*x[2]*x[3] - x[3]*x[4] - x[4]*x[5]
        - T(2)*x[5]*x[6] - x[6]*x[7] - T(2)*x[7]*x[8])
    ExaModels.@add_con(c, -x[1] - x[2] + T(1); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[3] - x[4] + T(1); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[5] - x[6] + T(1); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[7] - x[8] + T(1); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[1] - x[3] - x[5] - x[7] + T(2); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[2] - x[4] - x[6] - x[8] + T(2); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, T(2)*x[1] + x[3] - x[7]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, T(5)*x[1] + T(3)*x[3] - T(3)*x[5] - x[7]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[2] - x[4] - T(3)*x[6] - T(5)*x[8]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[2] - T(3)*x[6] - T(2)*x[8]; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
