# NONSCOMP: extended Rosenbrock (nonseparable version) with bounds (N=5000 variables)
# minimize (x[1] - 1)^2 + sum_{i=2}^{N} 4*(x[i] - x[i-1]^2)^2
# Bounds: lvar = -100 (all), uvar = 100 (all); odd-indexed variables also have lvar = 1.0
# Start: x[i] = 3.0 for all i
# Source: M. Lescrenier (1989); SIF input: Ph. Toint, May 1990
#
# SIF structure:
#   SQ(1): L2 group type (default), v = x[1] - 1.0 → (x[1]-1)^2
#   SQ(i) for i=2..N: L2 group type, scale=0.25, element ELA(i)=-X(i-1)^2
#     v = x[i] + (-x[i-1]^2) = x[i] - x[i-1]^2 → v^2/0.25 = 4*(x[i]-x[i-1]^2)^2
function nonscomp_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    # lvar: odd i → 1.0, even i → -100.0; uvar: 100.0 for all
    lvar = T[isodd(i) ? 1.0 : -100.0 for i in 1:N]
    uvar = fill(T(100.0), N)
    start = fill(T(3.0), N)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)
    # SQ(1): (x[1] - 1)^2
    ExaModels.@add_obj(c, (x[1] - T(1))^2)
    # SQ(i) for i=2..N: 4*(x[i] - x[i-1]^2)^2
    ExaModels.@add_obj(c, T(4) * (x[i] - x[i-1]^2)^2 for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end
