# CONGIGMZ: Constrained nonlinear minmax problem (3 variables, 5 constraints)
# minimize  Z
# G constraints (>= 0):
#   C1: Z + 5*X1 - X2 >= 0
#   C2: Z - 4*X2 - X1^2 - X2^2 >= 0
#   C3: Z - 5*X1 - X2 >= 0
# L constraints (<= 0):
#   C4: X1 + X2 <= -10   (i.e. X1 + X2 - (-10) <= 0)
#   C5: 2*X1^2 - X2^2 <= -4  (i.e. 2*X1^2 - X2^2 - (-4) <= 0)
# All variables free; start: X1=2, X2=2, Z=2
function congigmz_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
                       start = T[2.0, 2.0, 2.0])
    # x[1]=X1, x[2]=X2, x[3]=Z
    ExaModels.@add_obj(c, x[3])
    # CUTEst ordering: linear groups first (C1, C4, C3), then nonlinear (C2, C5)
    # C1 (G, linear): Z + 5*X1 - X2 >= 0
    ExaModels.@add_con(c, x[3] + T(5) * x[1] - x[2]; lcon = T(0), ucon = T(Inf))
    # C4 (L, linear): X1 + X2 - (-10) <= 0  →  X1 + X2 + 10 <= 0
    ExaModels.@add_con(c, x[1] + x[2] - T(-10); lcon = T(-Inf), ucon = T(0))
    # C3 (G, linear): Z - 5*X1 - X2 >= 0
    ExaModels.@add_con(c, x[3] - T(5) * x[1] - x[2]; lcon = T(0), ucon = T(Inf))
    # C2 (G, nonlinear): Z - 4*X2 - X1^2 - X2^2 >= 0
    ExaModels.@add_con(c, x[3] - T(4) * x[2] - x[1]^2 - x[2]^2; lcon = T(0), ucon = T(Inf))
    # C5 (L, nonlinear): 2*X1^2 - X2^2 - (-4) <= 0  →  2*X1^2 - X2^2 + 4 <= 0
    ExaModels.@add_con(c, T(2) * x[1]^2 - x[2]^2 - T(-4); lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
