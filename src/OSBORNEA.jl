# OSBORNEA: Osborne first problem (5 variables, unconstrained least squares)
# 33 groups, each L2 type (no scale): contribution = v_i^2
# v_i = X1 + X2*exp(T_i*X4) + X3*exp(T_i*X5) - C_i
# where T_i = -(i-1)*10 for i = 1..33
# BOUNDS: FR DEFAULT → all variables free
function osbornea_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf],
                       start = T[0.5, 1.5, -1.0, 0.01, 0.02])

    # Observed data constants C(I) for I=1..33
    ExaModels.@add_par(c, C, T[
        0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850,
        0.818, 0.784, 0.751, 0.718, 0.685, 0.658, 0.628, 0.603,
        0.580, 0.558, 0.538, 0.522, 0.506, 0.490, 0.478, 0.467,
        0.457, 0.448, 0.438, 0.431, 0.424, 0.420, 0.414, 0.411, 0.406
    ])

    # T_i = -(i-1)*10 for i = 1..33 (precomputed as parameter array)
    ExaModels.@add_par(c, TI, T[-(i-1)*10 for i in 1:33])

    # v_i = X1 + X2*exp(T_i*X4) + X3*exp(T_i*X5) - C_i
    # objective contribution = v_i^2 (L2 group type, no scale)
    ExaModels.@add_obj(c,
        (x[1] + x[2] * exp(TI[i] * x[4]) + x[3] * exp(TI[i] * x[5]) - C[i])^2
        for i in 1:33)

    return ExaModels.ExaModel(c; kwargs...)
end
