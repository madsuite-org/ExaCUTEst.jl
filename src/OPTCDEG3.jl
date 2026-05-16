# OPTCDEG3: Optimal control problem - cubically damped spring-mass
# Same structure as OPTCDEG2 but:
#   - A1 = 0.1*DT (different objective weight)
#   - C(t) uses CUBE(y(t)) instead of SQUARE(y(t))
#   - x(T)=0 is fixed, y(T) is free
#   - Extra term 1000*y(T)^2 in objective
function optcdeg3_model(Tval = 1500; T = Float64, backend = nothing, kwargs...)
    DT = T(20.0 / Tval)
    A1 = T(0.1 * DT)
    SPRINGKM = T(0.02)
    DAMPING  = T(0.05)
    C1 = SPRINGKM * DT
    C2 = DAMPING * DT
    MINUSDT = -DT

    nvar = 3 * Tval + 2
    offx = 0
    offy = Tval + 1
    offu = 2 * Tval + 2

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)

    for t in 0:Tval-1
        lvar[offy + t + 1] = T(-1.0)
        lvar[offu + t + 1] = T(-0.2)
        uvar[offu + t + 1] = T(0.2)
    end

    # Fixed: x(0)=10, x(T)=0, y(0)=0; y(T) free
    lvar[offx + 1] = T(10.0); uvar[offx + 1] = T(10.0)
    lvar[offx + Tval + 1] = T(0.0); uvar[offx + Tval + 1] = T(0.0)
    lvar[offy + 1] = T(0.0); uvar[offy + 1] = T(0.0)

    start = fill(T(0.0), nvar)
    start[offx + 1] = T(10.0)
    for t in 1:Tval-1
        start[offy + t + 1] = T(-1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Objective: A1*sum x(t)^2 + 1000*y(T)^2
    ExaModels.@add_obj(c, A1 * x[offx + t + 1]^2 for t in 0:Tval)
    ExaModels.@add_obj(c, T(1000.0) * x[offy + Tval + 1]^2)

    # Constraints B(t)
    ExaModels.@add_con(c, x[offx + t + 2] - x[offx + t + 1] + MINUSDT * x[offy + t + 1]
                       for t in 0:Tval-1)

    # Constraints C(t): CUBE element on y
    ExaModels.@add_con(c, x[offy + t + 2] - x[offy + t + 1] + MINUSDT * x[offu + t + 1] +
                       C1 * x[offx + t + 1] + C2 * x[offy + t + 1]^3
                       for t in 0:Tval-1)

    return ExaModels.ExaModel(c; kwargs...)
end
