# ENGVAL2: Engval2 problem (3 variables, unconstrained)
# Variables X1, X2, X3, all free, start = [1.0, 2.0, 0.0]
# All groups use L2 group type: contribution = v^2
#
# G1: elements SQ(X1)+SQ(X2)+SQ(X3), constant=1 → v = x1^2+x2^2+x3^2-1
# G2: elements SQ(X1)+SQ(X2)+SSQ(X3,S=2), constant=1 → v = x1^2+x2^2+(x3-2)^2-1
# G3: linear X1+X2+X3, constant=1 → v = x1+x2+x3-1
# G4: linear X1+X2-X3, constant=-1 → v = x1+x2-x3+1
# G5: elements 3*SQ(X2)+ELT(X1,X3), constant=36
#   ELT(X,Y): TWOW=5*Y-X+1, F=X^3+TWOW^2
#   → v = 3*x2^2 + x1^3 + (5*x3-x1+1)^2 - 36
function engval2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf), start = T[1.0, 2.0, 0.0])
    # G1: (x1^2 + x2^2 + x3^2 - 1)^2
    ExaModels.@add_obj(c, (x[1]^2 + x[2]^2 + x[3]^2 - 1)^2)
    # G2: (x1^2 + x2^2 + (x3 - 2)^2 - 1)^2
    ExaModels.@add_obj(c, (x[1]^2 + x[2]^2 + (x[3] - 2)^2 - 1)^2)
    # G3: (x1 + x2 + x3 - 1)^2
    ExaModels.@add_obj(c, (x[1] + x[2] + x[3] - 1)^2)
    # G4: (x1 + x2 - x3 + 1)^2
    ExaModels.@add_obj(c, (x[1] + x[2] - x[3] + 1)^2)
    # G5: (3*x2^2 + x1^3 + (5*x3 - x1 + 1)^2 - 36)^2
    ExaModels.@add_obj(c, (3 * x[2]^2 + x[1]^3 + (5 * x[3] - x[1] + 1)^2 - 36)^2)
    return ExaModels.ExaModel(c; kwargs...)
end
