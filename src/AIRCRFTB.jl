# AIRCRFTB: Aircraft stability least squares (Rheinboldt), minimization form
# 8 variables: same as AIRCRFTA but ELEVATOR=-0.05, AILERON=0.1, RUDDERDF=0 fixed
# Objective = G1^2 + G2^2 + G3^2 + G4^2 + G5^2 (L2 group type)
# No constraints
# Element 2PR(X,Y) = X*Y; L2 group: F(v) = v^2
function aircrftb_model(; T = Float64, backend = nothing, kwargs...)
    lvar = fill(T(-Inf), 8)
    uvar = fill(T(Inf), 8)
    lvar[6] = uvar[6] = T(-0.05)  # ELEVATOR fixed
    lvar[7] = uvar[7] = T(0.1)    # AILERON fixed
    lvar[8] = uvar[8] = T(0.0)    # RUDDERDF fixed
    start = T[T(0), T(0), T(0), T(0), T(0), T(-0.05), T(0.1), T(0)]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 8; lvar = lvar, uvar = uvar, start = start)
    # G1^2
    ExaModels.@add_obj(c, let v =
        T(-3.933)*x[1] + T(0.107)*x[2] + T(0.126)*x[3] - T(9.99)*x[5] -
        T(45.83)*x[7] - T(7.64)*x[8] +
        T(-0.727)*x[2]*x[3] + T(8.39)*x[3]*x[4] - T(684.4)*x[4]*x[5] +
        T(63.5)*x[2]*x[4]
        v^2
    end)
    # G2^2
    ExaModels.@add_obj(c, let v =
        T(-0.987)*x[2] - T(22.95)*x[4] - T(28.37)*x[6] +
        T(0.949)*x[1]*x[3] + T(0.173)*x[1]*x[5]
        v^2
    end)
    # G3^2
    ExaModels.@add_obj(c, let v =
        T(0.002)*x[1] - T(0.235)*x[3] + T(5.67)*x[5] - T(0.921)*x[7] -
        T(6.51)*x[8] +
        T(-0.716)*x[1]*x[2] - T(1.578)*x[1]*x[4] + T(1.132)*x[2]*x[4]
        v^2
    end)
    # G4^2
    ExaModels.@add_obj(c, let v = x[2] - x[4] - T(1.168)*x[6] - x[1]*x[5]; v^2 end)
    # G5^2
    ExaModels.@add_obj(c, let v = -x[3] - T(0.196)*x[5] - T(0.0071)*x[7] + x[1]*x[4]; v^2 end)
    return ExaModels.ExaModel(c; kwargs...)
end
