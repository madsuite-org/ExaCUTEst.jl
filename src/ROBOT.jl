# ROBOT: 7-link redundant robot arm in 2D
# Minimize sum of squared angular displacements from initial to final position
# 14 variables: TH1..TH7 (final angles), TH1I..TH7I (initial, fixed at 0)
# 2 equality constraints: x-position = 4, y-position = 4
# Each link has unit length (link 7 has length 0.5)
# ISQ element: U = TH - THi, F = U^2 -> objective sum (THk - THkI)^2
# COSTH element: F = cos(THETAC)  -- used for x-constraint
# SINTH element: F = sin(THETAS)  -- used for y-constraint
# CONSTR1 = sum_{k=1..7} cos(TH_k) [with last link coeff 0.5] = XPOS=4
# CONSTR2 = sum_{k=1..7} sin(TH_k) [with last link coeff 0.5] = YPOS=4
function robot_model(; T = Float64, backend = nothing, kwargs...)
    TH_in  = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # initial angles (all fixed 0)
    HIGH   = T(2.356194)
    DOWN   = T(-2.356194)
    XPOS   = T(4.0)
    YPOS   = T(4.0)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # Variables: TH1..TH7 (free in [DOWN, HIGH]), TH1I..TH7I (fixed at 0)
    ExaModels.@add_var(c, x, 14;
                       lvar = T[DOWN, DOWN, DOWN, DOWN, DOWN, DOWN, DOWN,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       uvar = T[HIGH, HIGH, HIGH, HIGH, HIGH, HIGH, HIGH,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       start = T(0))
    # Objective: sum_{k=1..7} (TH_k - TH_kI)^2
    # TH_k = x[k], TH_kI = x[k+7] (fixed at 0)
    ExaModels.@add_obj(c, (x[k] - x[k + 7])^2 for k in 1:7)
    # CONSTR1 (E type): sum_{k=1..6} cos(TH_k) + 0.5*cos(TH_7) = XPOS
    # All links except last have coeff 1 in GROUP USES
    ExaModels.@add_con(c,
        cos(x[1]) + cos(x[2]) + cos(x[3]) + cos(x[4]) +
        cos(x[5]) + cos(x[6]) + T(0.5) * cos(x[7]) - XPOS)
    # CONSTR2 (E type): sum_{k=1..6} sin(TH_k) + 0.5*sin(TH_7) = YPOS
    ExaModels.@add_con(c,
        sin(x[1]) + sin(x[2]) + sin(x[3]) + sin(x[4]) +
        sin(x[5]) + sin(x[6]) + T(0.5) * sin(x[7]) - YPOS)
    return ExaModels.ExaModel(c; kwargs...)
end
