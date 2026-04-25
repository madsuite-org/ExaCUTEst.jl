# BIGGSC4: Quadratic test problem with 4 variables and 7 constraints
# Source: Bartholomew-Biggs and Hernandez, 1992; SIF: Ph. Toint, April 1992
#
# Variables: X1..X4, lvar=0, uvar=5, start=0
# Objective (N type): -X1*X3 - X2*X4  (2PR elements)
# Constraints (G type, all with RANGE=5):
#   C1: X1+X2-2.5, C2: X1+X3-2.5, C3: X1+X4-2.5
#   C4: X2+X3-2.0, C5: X2+X4-2.0, C6: X3+X4-1.5  →  lcon=0, ucon=5
#   C7: X1+X2+X3+X4-5.0 (no RANGE)  →  lcon=0, ucon=Inf
function biggsc4_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
        lvar  = fill(T(0), 4),
        uvar  = fill(T(5), 4),
        start = fill(T(0), 4))
    ExaModels.@add_obj(c, -x[1] * x[3] - x[2] * x[4])
    ExaModels.@add_con(c, x[1] + x[2] - T(2.5); lcon = T(0), ucon = T(5))
    ExaModels.@add_con(c, x[1] + x[3] - T(2.5); lcon = T(0), ucon = T(5))
    ExaModels.@add_con(c, x[1] + x[4] - T(2.5); lcon = T(0), ucon = T(5))
    ExaModels.@add_con(c, x[2] + x[3] - T(2.0); lcon = T(0), ucon = T(5))
    ExaModels.@add_con(c, x[2] + x[4] - T(2.0); lcon = T(0), ucon = T(5))
    ExaModels.@add_con(c, x[3] + x[4] - T(1.5); lcon = T(0), ucon = T(5))
    ExaModels.@add_con(c, x[1] + x[2] + x[3] + x[4] - T(5.0); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
