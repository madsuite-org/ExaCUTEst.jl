# OPTCDEG2: Optimal control problem - damped spring-mass
# Variables: x(t), y(t) for t=0..T; u(t) for t=0..T-1
# Constraints B(t): x(t+1) - x(t) - DT*y(t) = 0
#             C(t): y(t+1) - y(t) - DT*u(t) + C1*x(t) + C2*y(t)^2 = 0 (note: SQR on y)
# Objective: sum A1*x(t)^2 for t=0..T
# T=1500, variable layout: x(0..T) occupy indices 1..T+1, y(0..T): T+2..2T+2, u(0..T-1): 2T+3..3T+2
function optcdeg2_model(Tval = 1500; T = Float64, backend = nothing, kwargs...)
    DT = T(20.0 / Tval)
    A1 = T(0.5 * DT)
    SPRINGKM = T(0.02)
    DAMPING  = T(0.05)
    C1 = SPRINGKM * DT
    C2 = DAMPING * DT
    MINUSDT = -DT

    # Variable layout:
    # x[1..T+1]   = x(0..T) in SIF
    # x[T+2..2T+2] = y(0..T) in SIF
    # x[2T+3..3T+2] = u(0..T-1) in SIF
    nvar = 3 * Tval + 2
    offx = 0      # x(t) at index offx+t+1
    offy = Tval+1 # y(t) at index offy+t+1
    offu = 2*Tval+2 # u(t) at index offu+t+1

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)

    # Bounds on y and u (for t=0..T-1)
    for t in 0:Tval-1
        lvar[offy + t + 1] = T(-1.0)  # y >= -1
        lvar[offu + t + 1] = T(-0.2)  # u >= -0.2
        uvar[offu + t + 1] = T(0.2)   # u <= 0.2
    end

    # Fixed variables
    lvar[offx + 1] = T(10.0); uvar[offx + 1] = T(10.0)  # x(0) = 10
    # x(T) is free (XR)
    lvar[offy + 1] = T(0.0); uvar[offy + 1] = T(0.0)     # y(0) = 0
    lvar[offy + Tval + 1] = T(0.0); uvar[offy + Tval + 1] = T(0.0) # y(T) = 0

    # Start: y(t)=-1 for t=1..T-1
    start = fill(T(0.0), nvar)
    start[offx + 1] = T(10.0)
    for t in 1:Tval-1
        start[offy + t + 1] = T(-1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Objective: sum A1 * x(t)^2 for t=0..T
    ExaModels.@add_obj(c, A1 * x[offx + t + 1]^2 for t in 0:Tval)

    # Constraints B(t): x(t+1) - x(t) - DT*y(t) = 0 for t=0..T-1
    ExaModels.@add_con(c, x[offx + t + 2] - x[offx + t + 1] + MINUSDT * x[offy + t + 1]
                       for t in 0:Tval-1)

    # Constraints C(t): y(t+1) - y(t) - DT*u(t) + C1*x(t) + C2*y(t)^2 = 0
    ExaModels.@add_con(c, x[offy + t + 2] - x[offy + t + 1] + MINUSDT * x[offu + t + 1] +
                       C1 * x[offx + t + 1] + C2 * x[offy + t + 1]^2
                       for t in 0:Tval-1)

    return ExaModels.ExaModel(c; kwargs...)
end
