# MWRIGHT: 5 variables, 3 equality constraints
# Group types: L2(v) = v^2, L3(v) = v^3, L4(v) = v^4
# Objective groups (all N type, identity):
#   OBJ1: v = x1 (L2)
#   OBJ2: v = x1 - x2 (L2)
#   OBJ3: v = x2 - x3 (L3)
#   OBJ4: v = x3 - x4 (L4)
#   OBJ5: v = x4 - x5 (L4)
# minimize x1^2 + (x1-x2)^2 + (x2-x3)^3 + (x3-x4)^4 + (x4-x5)^4
# Constraints (E type):
#   C1: x1 + x2^2 + x3^2 = 3*sqrt(2) + 2
#   C2: x2 + x4 - x3^2 = 2*sqrt(2) - 2  (C2: XE C2 X3SQ -1.0 → -x3^2)
#   C3: x1*x5 = 2
# All variables free; start: x1=-1, x2=2, x3=1, x4=-2, x5=-2
function mwright_model(; T = Float64, backend = nothing, kwargs...)
    sq2 = sqrt(T(2))
    rhs1 = T(3) * sq2 + T(2)
    rhs2 = T(2) * sq2 - T(2)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T(-Inf), uvar = T(Inf),
                       start = T[-1.0, 2.0, 1.0, -2.0, -2.0])
    ExaModels.@add_obj(c, x[1]^2 + (x[1] - x[2])^2 + (x[2] - x[3])^3 +
                          (x[3] - x[4])^4 + (x[4] - x[5])^4)
    # C1: x1 + x2^2 + x3^2 - rhs1 = 0
    ExaModels.@add_con(c, x[1] + x[2]^2 + x[3]^2 - rhs1)
    # C2: x2 + x4 - x3^2 - rhs2 = 0
    ExaModels.@add_con(c, x[2] + x[4] - x[3]^2 - rhs2)
    # C3: x1*x5 - 2 = 0
    ExaModels.@add_con(c, x[1] * x[5] - T(2))
    return ExaModels.ExaModel(c; kwargs...)
end
