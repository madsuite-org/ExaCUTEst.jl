# ZECEVIC3: Zecevic problem 3 (2 variables, 2 inequality constraints)
# minimize 7*X1^2 + 3*X2^2 - 84*X1 - 24*X2 + 300
# subject to 1 - X1*X2 <= 0
#            X1^2 + X2^2 - 9 <= 0
# Bounds: 0 <= X1, X2 <= 10
# Source: Zecevic (1993), SIF: Gould, April 1993
function zecevic3_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[0.0, 0.0], uvar = T[10.0, 10.0],
                       start = T[0.1, -0.1])
    # OBJ: N group (identity), linear -84*X1 - 24*X2, elements 7*X1^2 + 3*X2^2, rhs=-300
    # v_OBJ = -84*X1 - 24*X2 + 7*X1^2 + 3*X2^2 - (-300) = 7*X1^2 + 3*X2^2 - 84*X1 - 24*X2 + 300
    ExaModels.@add_obj(c, T(7)*x[1]^2 + T(3)*x[2]^2 - T(84)*x[1] - T(24)*x[2] + T(300))
    # CON1: L group, element PROD(X1,X2)*(-1.0), rhs=-1.0
    # v_CON1 = 0 - (-1) + (-1)*X1*X2 = 1 - X1*X2
    ExaModels.@add_con(c, T(1) - x[1]*x[2]; lcon = T(-Inf), ucon = T(0))
    # CON2: L group, elements X1^2 + X2^2, rhs=9.0
    # v_CON2 = X1^2 + X2^2 - 9
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - T(9); lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
