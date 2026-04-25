# AIRCRFTA: Aircraft stability (Rheinboldt), feasibility problem
# 8 variables: ROLLRATE(1), PITCHRAT(2), YAWRATE(3), ATTCKANG(4), SSLIPANG(5),
#              ELEVATOR(6), AILERON(7), RUDDERDF(8)
# Variables 1-5 are free; 6=0.1 fixed (ELEVATOR), 7=0 fixed (AILERON), 8=0 fixed (RUDDER)
# No objective; 5 equality constraints
# Element 2PR(X,Y) = X*Y
function aircrfta_model(; T = Float64, backend = nothing, kwargs...)
    lvar = fill(T(-Inf), 8)
    uvar = fill(T(Inf), 8)
    lvar[6] = uvar[6] = T(0.1)   # ELEVATOR fixed
    lvar[7] = uvar[7] = T(0.0)   # AILERON fixed
    lvar[8] = uvar[8] = T(0.0)   # RUDDERDF fixed
    start = T[T(0), T(0), T(0), T(0), T(0), T(0.1), T(0), T(0)]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 8; lvar = lvar, uvar = uvar, start = start)
    # G1: -3.933x1+0.107x2+0.126x3-9.99x5-45.83x7-7.64x8
    #     -0.727*x2*x3 + 8.39*x3*x4 - 684.4*x4*x5 + 63.5*x2*x4 = 0
    ExaModels.@add_con(c,
        T(-3.933)*x[1] + T(0.107)*x[2] + T(0.126)*x[3] - T(9.99)*x[5] -
        T(45.83)*x[7] - T(7.64)*x[8] +
        T(-0.727)*x[2]*x[3] + T(8.39)*x[3]*x[4] - T(684.4)*x[4]*x[5] +
        T(63.5)*x[2]*x[4])
    # G2: -0.987x2-22.95x4-28.37x6 + 0.949*x1*x3 + 0.173*x1*x5 = 0
    ExaModels.@add_con(c,
        T(-0.987)*x[2] - T(22.95)*x[4] - T(28.37)*x[6] +
        T(0.949)*x[1]*x[3] + T(0.173)*x[1]*x[5])
    # G3: 0.002x1-0.235x3+5.67x5-0.921x7-6.51x8
    #     -0.716*x1*x2 - 1.578*x1*x4 + 1.132*x2*x4 = 0
    ExaModels.@add_con(c,
        T(0.002)*x[1] - T(0.235)*x[3] + T(5.67)*x[5] - T(0.921)*x[7] -
        T(6.51)*x[8] +
        T(-0.716)*x[1]*x[2] - T(1.578)*x[1]*x[4] + T(1.132)*x[2]*x[4])
    # G4: x2-x4-1.168x6 - x1*x5 = 0
    ExaModels.@add_con(c, x[2] - x[4] - T(1.168)*x[6] - x[1]*x[5])
    # G5: -x3-0.196x5-0.0071x7 + x1*x4 = 0
    ExaModels.@add_con(c, -x[3] - T(0.196)*x[5] - T(0.0071)*x[7] + x[1]*x[4])
    return ExaModels.ExaModel(c; kwargs...)
end
