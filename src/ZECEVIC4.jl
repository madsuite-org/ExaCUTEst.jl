# ZECEVIC4: Zecevic problem 4 (2 variables, 2 inequality constraints)
# minimize 6*X1^2 + X2^2 - 60*X1 - 8*X2 + 166
# subject to -X1 - X2 + 3 <= 0  (CON2, linear only - appears first in CUTEst)
#            -X1 - X2 + X1*X2 <= 0  (CON1, with element)
# Bounds: 0 <= X1, X2 <= 10
# Source: Zecevic (1993), SIF: Gould, April 1993
# Note: CUTEst places the purely-linear group (CON2) before the group with elements (CON1)
function zecevic4_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[0.0, 0.0], uvar = T[10.0, 10.0],
                       start = T[0.1, -0.1])
    # OBJ: N group (identity), linear -60*X1 - 8*X2, elements 6*X1^2 + X2^2, rhs=-166
    # v_OBJ = -60*X1 - 8*X2 + 6*X1^2 + X2^2 + 166
    ExaModels.@add_obj(c, T(6)*x[1]^2 + x[2]^2 - T(60)*x[1] - T(8)*x[2] + T(166))
    # CON2 (listed first by CUTEst): L group, linear -X1 - X2, rhs=-3
    # v_CON2 = -X1 - X2 - (-3) = 3 - X1 - X2
    ExaModels.@add_con(c, T(3) - x[1] - x[2]; lcon = T(-Inf), ucon = T(0))
    # CON1 (listed second by CUTEst): L group, linear -X1 - X2, element PROD(X1,X2)*1.0, rhs=0
    # v_CON1 = -X1 - X2 + X1*X2
    ExaModels.@add_con(c, -x[1] - x[2] + x[1]*x[2]; lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
