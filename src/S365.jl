# S365: Walukiewicz problem (7 variables, 5 inequality constraints)
# Source: Schittkowski "More Test Examples for NLP Codes", problem 365
#
# Element types (expanded):
#   2PR(X,Y): F = X*Y
#   ISQ(X,Y): U=X-Y, F=U^2 = (X-Y)^2
#   PX(X,Y):  F = sqrt(X^2) + Y^2  = |X| + Y^2
#   QX(X,Y,Z): U=X, V=Y-Z, F = sqrt(U^2) + V^2 = |X| + (Y-Z)^2
#
# Objective: OBJ = x1*x3 (from X1X3 element)
#
# Constraints (G type, >= 0, rhs from CONSTANTS: C1 rhs=4):
#   C1: E2 + E3 = (X4-X6)^2 + (X5-X7)^2 >= 4
#   C2: X3X4 - X2X5 - EP = X3*X4 - X2*X5 - (|X2| + X3^2) >= 0
#   C3: X3X6 - X2X7 - EP = X3*X6 - X2*X7 - (|X2| + X3^2) >= 0
#   C4: X1X3 + X2X5 - X1X5 - X3X4 - EQ
#       = X1*X3 + X2*X5 - X1*X5 - X3*X4 - (|X3| + (X2-X1)^2) >= 0
#   C5: X1X3 + X2X7 - X1X7 - X3X6 - EQ
#       = X1*X3 + X2*X7 - X1*X7 - X3*X6 - (|X3| + (X2-X1)^2) >= 0
#
# Bounds: FR 'DEFAULT'; XL: x1>=0, x3>=0, x5>=1, x7>=1
# Start: x1=3, x2=0, x3=2, x4=-1.5, x5=1.5, x6=5, x7=0
function s365_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 7;
                       lvar = T[0.0, -Inf, 0.0, -Inf, 1.0, -Inf, 1.0],
                       uvar = T[Inf, Inf, Inf, Inf, Inf, Inf, Inf],
                       start = T[3.0, 0.0, 2.0, -1.5, 1.5, 5.0, 0.0])
    # Objective: x1*x3
    ExaModels.@add_obj(c, x[1] * x[3])
    # C1: (x4-x6)^2 + (x5-x7)^2 - 4 >= 0
    ExaModels.@add_con(c, (x[4] - x[6])^2 + (x[5] - x[7])^2 - T(4);
                       lcon = T(0), ucon = T(Inf))
    # C2: x3*x4 - x2*x5 - |x2| - x3^2 >= 0
    ExaModels.@add_con(c, x[3] * x[4] - x[2] * x[5] - sqrt(x[2]^2) - x[3]^2;
                       lcon = T(0), ucon = T(Inf))
    # C3: x3*x6 - x2*x7 - |x2| - x3^2 >= 0
    ExaModels.@add_con(c, x[3] * x[6] - x[2] * x[7] - sqrt(x[2]^2) - x[3]^2;
                       lcon = T(0), ucon = T(Inf))
    # C4: x1*x3 + x2*x5 - x1*x5 - x3*x4 - |x3| - (x2-x1)^2 >= 0
    ExaModels.@add_con(c,
        x[1] * x[3] + x[2] * x[5] - x[1] * x[5] - x[3] * x[4] -
        sqrt(x[3]^2) - (x[2] - x[1])^2;
        lcon = T(0), ucon = T(Inf))
    # C5: x1*x3 + x2*x7 - x1*x7 - x3*x6 - |x3| - (x2-x1)^2 >= 0
    ExaModels.@add_con(c,
        x[1] * x[3] + x[2] * x[7] - x[1] * x[7] - x[3] * x[6] -
        sqrt(x[3]^2) - (x[2] - x[1])^2;
        lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
