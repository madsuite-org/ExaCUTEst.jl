# HATFLDH: Hatfield polytechnic test problem (4 variables, 7 constraints)
# Nonlinear objective with linear constraints.
#
# Variables: X1..X4; lvar=0 (SIF default), uvar=5 (XU DEFAULT 5.0)
# start=[1, 5, 5, 1]
#
# Objective (N type, identity):
#   OBJ = -1.0*A1 - 1.0*A2  where A1 = X1*X3, A2 = X2*X4
#   so OBJ = -X1*X3 - X2*X4
#
# Constraints (G type ≥ 0, linear; RANGES add upper bound 0+range=range):
#   C1: X1 + X2 - 2.5,  RANGE=5.0  → lcon=0, ucon=5.0
#   C2: X1 + X3 - 2.5,  RANGE=5.0  → lcon=0, ucon=5.0
#   C3: X1 + X4 - 2.5,  RANGE=5.0  → lcon=0, ucon=5.0
#   C4: X2 + X3 - 2.0,  RANGE=5.0  → lcon=0, ucon=5.0
#   C5: X2 + X4 - 2.0,  RANGE=5.0  → lcon=0, ucon=5.0
#   C6: X3 + X4 - 1.5,  RANGE=5.0  → lcon=0, ucon=5.0
#   C7: X1 + X2 + X3 + X4 - 5.0,  no RANGE → lcon=0, ucon=Inf
function hatfldh_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
                       lvar  = T[0.0, 0.0, 0.0, 0.0],
                       uvar  = T[5.0, 5.0, 5.0, 5.0],
                       start = T[1.0, 5.0, 5.0, 1.0])

    # Objective: -X1*X3 - X2*X4
    ExaModels.@add_obj(c, -x[1] * x[3] - x[2] * x[4])

    # C1: X1 + X2 - 2.5 ∈ [0, 5]
    ExaModels.@add_con(c, x[1] + x[2] - T(2.5); lcon = T(0), ucon = T(5.0))
    # C2: X1 + X3 - 2.5 ∈ [0, 5]
    ExaModels.@add_con(c, x[1] + x[3] - T(2.5); lcon = T(0), ucon = T(5.0))
    # C3: X1 + X4 - 2.5 ∈ [0, 5]
    ExaModels.@add_con(c, x[1] + x[4] - T(2.5); lcon = T(0), ucon = T(5.0))
    # C4: X2 + X3 - 2.0 ∈ [0, 5]
    ExaModels.@add_con(c, x[2] + x[3] - T(2.0); lcon = T(0), ucon = T(5.0))
    # C5: X2 + X4 - 2.0 ∈ [0, 5]
    ExaModels.@add_con(c, x[2] + x[4] - T(2.0); lcon = T(0), ucon = T(5.0))
    # C6: X3 + X4 - 1.5 ∈ [0, 5]
    ExaModels.@add_con(c, x[3] + x[4] - T(1.5); lcon = T(0), ucon = T(5.0))
    # C7: X1 + X2 + X3 + X4 - 5.0 ≥ 0 (no RANGE)
    ExaModels.@add_con(c, x[1] + x[2] + x[3] + x[4] - T(5.0); lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
