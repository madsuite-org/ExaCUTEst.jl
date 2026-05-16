# LSNNODOC: 5 variables, 4 equality constraints, unconstrained objective
# minimize z2*(x1+x3)*exp(x1+x3) + x3^2*x4^2 + (x3-x5)^2
# EXPO element: T=X+Y, TZ=Z, F=TZ*exp(T), so with X=X1,Y=X3,Z=X2: X2*(X1+X3)*exp(X1+X3)
# SQ2 element: F=X^2*Y^2, with X=X3,Y=X4: X3^2*X4^2
# ISQ element: F=(X-Y)^2, with X=X3,Y=X5: (X3-X5)^2
# Constraints (E type, linear):
#  C1: x1 + x2 = 10
#  C2: -x1 + x3 - x3 + x4 = 0 → actually C2: -x1 - x3 + x4 = 0 → rhs=0
#  but CONSTANTS: C1=10, C4=-10
#  C1: x1 + x2 - 10 = 0
#  C2: -x1 - x3 + x4 = 0
#  C3: -x2 + x3 + x5 = 0
#  C4: -x4 - x5 = -10 → -x4 - x5 + 10 = 0
# Bounds: x1 in [2,4], x2 in [6,8], x3 in [0,5], x4,x5 free
function lsnnodoc_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5;
                       lvar = T[2.0, 6.0, 0.0, -Inf, -Inf],
                       uvar = T[4.0, 8.0, 5.0, Inf, Inf],
                       start = T[4.0, 6.0, 2.0, 6.0, 4.0])
    # Objective: E1 = X2*(X1+X3)*exp(X1+X3), E2 = X3^2*X4^2, E3 = (X3-X5)^2
    ExaModels.@add_obj(c, x[2] * (x[1] + x[3]) * exp(x[1] + x[3]) +
                          x[3]^2 * x[4]^2 + (x[3] - x[5])^2)
    # C1: x1 + x2 - 10 = 0
    ExaModels.@add_con(c, x[1] + x[2] - T(10))
    # C2: -x1 - x3 + x4 = 0
    ExaModels.@add_con(c, -x[1] - x[3] + x[4])
    # C3: -x2 + x3 + x5 = 0
    ExaModels.@add_con(c, -x[2] + x[3] + x[5])
    # C4: -x4 - x5 + 10 = 0
    ExaModels.@add_con(c, -x[4] - x[5] + T(10))
    return ExaModels.ExaModel(c; kwargs...)
end
