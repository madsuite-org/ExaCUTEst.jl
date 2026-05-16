# LOGHAIRY: 2 variables, unconstrained
# One group FURCUP of type LOG: F(g) = log((100+g)/100)
# Group value: v = HLENGTH*FUR + CSLOPE*DBOWL + CSLOPE*1BOWL
# FUR = sin(DENS*X1)^2 * cos(DENS*X2)^2, DENS=7
# DBOWL: IV V = X1-X2; F = sqrt(SMOOTH + V^2), SMOOTH=0.01
# 1BOWL: F = sqrt(SMOOTH + X1^2), SMOOTH=0.01
# HLENGTH=30, CSLOPE=100, SMOOTH=0.01, DENS=7
# LOG group: F(g) = log((S+g)/S) where S=100
# Start: X1=-500, X2=-700
function loghairy_model(; T = Float64, backend = nothing, kwargs...)
    HLENGTH = T(30)
    CSLOPE = T(100)
    SMOOTH = T(0.01)
    DENS = T(7)
    S = T(100)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
                       start = T[-500.0, -700.0])
    # FUR = sin(DENS*x1)^2 * cos(DENS*x2)^2
    # DBOWL = sqrt(SMOOTH + (x1-x2)^2)
    # 1BOWL = sqrt(SMOOTH + x1^2)
    # v = HLENGTH*FUR + CSLOPE*DBOWL + CSLOPE*1BOWL
    # F(v) = log((S+v)/S) = log(1 + v/S)
    ExaModels.@add_obj(c,
        log((S + HLENGTH * sin(DENS * x[1])^2 * cos(DENS * x[2])^2 +
             CSLOPE * sqrt(SMOOTH + (x[1] - x[2])^2) +
             CSLOPE * sqrt(SMOOTH + x[1]^2)) / S))
    return ExaModels.ExaModel(c; kwargs...)
end
