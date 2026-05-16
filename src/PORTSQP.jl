# PORTSQP: Convex quadratic portfolio-type QP
# N vars x_i >= 0 (no upper bound)
# Objective: sum_i (x_i - C_i)^2 where C_i = (2i-N)/N
#   L2 group type: (x_i - C_i)^2
# Constraint E: sum_i x_i = 1
function portsqp_model(N = 100000; T = Float64, backend = nothing, kwargs...)
    RN = T(N)
    lvar = zeros(T, N)
    uvar = fill(T(Inf), N)
    start = fill(T(0.5), N)

    Cvals = T[(T(2) * T(i) - RN) / RN for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, C, Cvals)

    # Objective: sum_i (x_i - (2i-N)/N)^2
    ExaModels.@add_obj(c, (x[i] - C[i])^2 for i in 1:N)

    # Constraint: sum_i x_i = 1
    ExaModels.@add_con(c, sum(x[i] for i in 1:N) - T(1.0))

    return ExaModels.ExaModel(c; kwargs...)
end
