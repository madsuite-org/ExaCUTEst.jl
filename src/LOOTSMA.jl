# LOOTSMA: 3 variables, 2 inequality constraints
# minimize 11*x1 + x1^3 - 6*x1^2 + x3
# subject to -x1^2 - x2^2 + x3^2 >= 0
#            x1^2 + x2^2 + x3^2 >= 4  (rhs=4)
# Bounds: x3 <= 5, x1>=0, x2>=0 (SIF default lvar=0, uvar=5 for x3)
# SIF: Xc1 G type: -x1^2 - x2^2 + x3^2 >= 0
#       Xc2 G type: x1^2 + x2^2 + x3^2 >= 4
function lootsma_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[0.0, 0.0, 0.0], uvar = T[Inf, Inf, 5.0],
                       start = T[1.0, 1.0, -3.0])
    # Objective: N OBJ = 11*x1 + x3; elements: X1CB (x1^3), X1SQ coeff -6 (x1^2)
    ExaModels.@add_obj(c, T(11) * x[1] + x[1]^3 - T(6) * x[1]^2 + x[3])
    # Xc1 G: -x1^2 - x2^2 + x3^2 >= 0
    ExaModels.@add_con(c, -x[1]^2 - x[2]^2 + x[3]^2; lcon = T(0), ucon = T(Inf))
    # Xc2 G: x1^2 + x2^2 + x3^2 >= 4
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 + x[3]^2 - T(4); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
