# MATRIX2: 6 variables, 2 inequality constraints
# Variables: X11, X12, X22, Y11, Y12, Y22 (2x2 symmetric matrix entries)
# Minimize (X11-Y11)^2 + 2*(X12-Y12)^2 + (X22-Y22)^2
# subject to X11*X22 - X12^2 >= 0   (X posdef)
#            Y11*Y22 - Y12^2 <= 0   (Y negdef)
# Bounds: X12, Y12 free; Y11, Y22 <= 0; X11, X22 >= 0 (default)
# XR = free, XM = upper bounded by 0 (same as UP 0)
# Start: all 1.0
function matrix2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # Variables: [X11, X12, X22, Y11, Y12, Y22]
    # X11>=0, X12 free, X22>=0, Y11<=0, Y12 free, Y22<=0
    ExaModels.@add_var(c, x, 6;
                       lvar = T[0.0, -Inf, 0.0, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, 0.0, Inf, 0.0],
                       start = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # x[1]=X11, x[2]=X12, x[3]=X22, x[4]=Y11, x[5]=Y12, x[6]=Y22
    # Objective: (X11-Y11)^2 + 2*(X12-Y12)^2 + (X22-Y22)^2
    ExaModels.@add_obj(c, (x[1] - x[4])^2 + T(2) * (x[2] - x[5])^2 + (x[3] - x[6])^2)
    # XPOSDEF: X11*X22 - X12^2 >= 0
    ExaModels.@add_con(c, x[1] * x[3] - x[2]^2; lcon = T(0), ucon = T(Inf))
    # YNEGDEF: Y11*Y22 - Y12^2 <= 0
    ExaModels.@add_con(c, x[4] * x[6] - x[5]^2; lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
