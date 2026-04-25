# OSBORNEB: Osborne second problem (11 variables, unconstrained least squares)
# 65 groups, all L2 type (no scale): contribution = v_i^2
# v_i = X1*exp(-ti*X5) + X2*exp(-(ti-X9)^2*X6)
#       + X3*exp(-(ti-X10)^2*X7) + X4*exp(-(ti-X11)^2*X8) - C_i
#
# KEY: In the SIF, the DO loop uses:
#   IA I-1  I  1    (note: +1, not -1!)
#   RI RI-1  I-1    → RI-1 = float(I + 1)
#   RM TI    RI-1  0.1  → TI = (I+1)*0.1
# So for I=1..65: TI = 0.2, 0.3, ..., 6.6 (not 0.0, 0.1, ..., 6.4)
#
# BOUNDS: FR DEFAULT → all variables free
function osborneb_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 11;
        lvar  = T[-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf],
        uvar  = T[ Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf],
        start = T[1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5])

    # Observed data constants C(I) for I=1..65
    ExaModels.@add_par(c, C, T[
        1.366, 1.191, 1.112, 1.013, 0.991,
        0.885, 0.831, 0.847, 0.786, 0.725,
        0.746, 0.679, 0.608, 0.655, 0.616,
        0.606, 0.602, 0.626, 0.651, 0.724,
        0.649, 0.649, 0.694, 0.644, 0.624,
        0.661, 0.612, 0.558, 0.533, 0.495,
        0.500, 0.423, 0.395, 0.375, 0.372,
        0.391, 0.396, 0.405, 0.428, 0.429,
        0.523, 0.562, 0.607, 0.653, 0.672,
        0.708, 0.633, 0.668, 0.645, 0.632,
        0.591, 0.559, 0.597, 0.625, 0.739,
        0.710, 0.729, 0.720, 0.636, 0.581,
        0.428, 0.292, 0.162, 0.098, 0.054
    ])

    # ti = (i+1)*0.1 for i = 1..65 → 0.2, 0.3, ..., 6.6
    # (SIF uses IA I-1 I 1, which gives I-1 = I+1, so TI = (I+1)*0.1)
    ExaModels.@add_par(c, TI, T[(i+1)*0.1 for i in 1:65])

    # v_i = PEXP(X1,X5;TI) + PEXP3(X2,X9,X6;TI) + PEXP3(X3,X10,X7;TI) + PEXP3(X4,X11,X8;TI) - C_i
    # PEXP(V1,V2;T)      = V1 * exp(-T*V2)
    # PEXP3(V1,V2,V3;T3) = V1 * exp(-(T3-V2)^2 * V3)
    # contribution = v_i^2 (L2 group type, no scale)
    ExaModels.@add_obj(c,
        (  x[1]  * exp(-TI[i] * x[5])
         + x[2]  * exp(-(TI[i] - x[9])^2  * x[6])
         + x[3]  * exp(-(TI[i] - x[10])^2 * x[7])
         + x[4]  * exp(-(TI[i] - x[11])^2 * x[8])
         - C[i])^2
        for i in 1:65)

    return ExaModels.ExaModel(c; kwargs...)
end
