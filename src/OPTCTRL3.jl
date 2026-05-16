# OPTCTRL3: Optimal control problem with penalty on u
# Variables: x(t),y(t) for t=0..T; u(t) for t=0..T-1
# Constraints (linear!):
#   B(t): x(t+1) - x(t) - 0.2*y(t) = 0
#   C(t): y(t+1) - y(t) + 0.004*x(t) - 0.2*u(t) + 0.01*y(t)^2 = 0
# Objective: sum 0.5*x(t)^2 + TDP_mu*(u(t)-0.2)^2 for relevant terms
function optctrl3_model(Tval = 1500; T = Float64, backend = nothing, kwargs...)
    TDP_mu = T(1.0e6)

    nvar = 3 * Tval + 2
    offx = 0
    offy = Tval + 1
    offu = 2 * Tval + 2

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)

    # Fixed: x(0)=10, y(0)=0, y(T)=0; x(T) free
    lvar[offx + 1] = T(10.0); uvar[offx + 1] = T(10.0)
    lvar[offy + 1] = T(0.0);  uvar[offy + 1] = T(0.0)
    lvar[offy + Tval + 1] = T(0.0); uvar[offy + Tval + 1] = T(0.0)

    start = fill(T(0.0), nvar)
    start[offx + 1] = T(10.0)
    for t in 1:Tval-1
        start[offy + t + 1] = T(-1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Objective: 0.5*x(t)^2 for t=0..T
    ExaModels.@add_obj(c, T(0.5) * x[offx + t + 1]^2 for t in 0:Tval)
    # Penalty: TDP_mu * (u(t) - 0.2)^2 for t=1..T-1
    ExaModels.@add_obj(c, TDP_mu * (x[offu + t + 1] - T(0.2))^2 for t in 1:Tval-1)
    # C(t): +0.01*y(t)^2 (SQR element with coeff 0.01)
    # Also included in constraints

    # Constraints B(t): x(t+1) - x(t) - 0.2*y(t) = 0
    ExaModels.@add_con(c, x[offx + t + 2] - x[offx + t + 1] - T(0.2) * x[offy + t + 1]
                       for t in 0:Tval-1)
    # Constraints C(t): y(t+1) - y(t) + 0.004*x(t) - 0.2*u(t) + 0.01*y(t)^2 = 0
    ExaModels.@add_con(c, x[offy + t + 2] - x[offy + t + 1] + T(0.004) * x[offx + t + 1] -
                       T(0.2) * x[offu + t + 1] + T(0.01) * x[offy + t + 1]^2
                       for t in 0:Tval-1)

    return ExaModels.ExaModel(c; kwargs...)
end
