# GOTTFR: Gottfried-Frisch problem (2 variables, 2 equality constraints)
# Source: Buckley (1989), problem 208; Sisser (1981)
# SIF: classification NQR2-AN-2-2
#
# Variables: X1, X2 (both free), start = 0.5
#
# Element ET1(X,Y):  FA = X + 3*Y,  FB = 1 - X
#   F = -0.1136 * FA * FB = -0.1136*(X + 3*Y)*(1 - X)
#
# Element ET2(X,Y):  FA = 2*X - Y,  FB = 1 - Y
#   F = 7.5 * FA * FB = 7.5*(2*X - Y)*(1 - Y)
#
# Groups (E type = equality, identity group function):
#   G1 = X1 + E1(X1,X2) = 0
#      = X1 - 0.1136*(X1 + 3*X2)*(1 - X1) = 0
#   G2 = X2 + E2(X1,X2) = 0
#      = X2 + 7.5*(2*X1 - X2)*(1 - X2) = 0
function gottfr_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T(0.5))
    # No objective (pure feasibility / equality-constrained problem)
    ExaModels.@add_obj(c, T(0) * x[1])
    # G1 = X1 + ET1(X1,X2) = X1 - 0.1136*(X1+3*X2)*(1-X1) = 0
    ExaModels.@add_con(c, x[1] - T(0.1136) * (x[1] + T(3) * x[2]) * (T(1) - x[1]))
    # G2 = X2 + ET2(X1,X2) = X2 + 7.5*(2*X1-X2)*(1-X2) = 0
    ExaModels.@add_con(c, x[2] + T(7.5) * (T(2) * x[1] - x[2]) * (T(1) - x[2]))
    return ExaModels.ExaModel(c; kwargs...)
end
