# POWELL20: Degenerate convex QP - generalization of Powell problem 20
# Minimize 0.5*sum x_i^2
# Subject to: x(k+1) - x(k) >= (-1)^k * k - 0.5  for k=1..N-1
#             x(1) - x(N) >= N - 0.5
# SQ element: F = 0.5*X^2
# PLUSMINUS starts at 1, flips sign each k:
#   k=1: rhs = 1*1 - 0.5 = 0.5; sign=-1 → k=2: rhs = -1*2 - 0.5 = -2.5
#   k=2: rhs = -2 - 0.5 = -2.5; sign=+1 → k=3: rhs = 3 - 0.5 = 2.5
# General: rhs(k) = (-1)^(k+1) * k - 0.5
function powell20_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf), N)
    start = zeros(T, N)

    # Compute RHS values
    rhs = Vector{T}(undef, N)
    plusminus = T(1.0)
    for k in 1:N-1
        rhs[k] = plusminus * T(k) - T(0.5)
        plusminus = -plusminus
    end
    rhs[N] = T(N - 1) - T(0.5)  # = N - 0.5 but SIF says RN-1/2 = N - 0.5... wait
    # SIF: N-1/2 = RN - 0.5 where RN = float(N), so N-0.5
    rhs[N] = T(N) - T(0.5)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # Objective: sum 0.5 * x[i]^2
    ExaModels.@add_obj(c, T(0.5) * x[i]^2 for i in 1:N)

    # Constraints CON(k): x(k+1) - x(k) - rhs(k) >= 0 for k=1..N-1
    ExaModels.@add_par(c, θ_rhs, rhs[1:N-1])
    ExaModels.@add_con(c, x[k+1] - x[k] - θ_rhs[k] for k in 1:N-1;
                       lcon = T(0.0), ucon = T(Inf))

    # CON(N): x(1) - x(N) >= N - 0.5
    ExaModels.@add_con(c, x[1] - x[N] - rhs[N]; lcon = T(0.0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
