# ENGVAL1: Engval1 problem (scalable, unconstrained)
# N variables (default N=5000), all free, start = 2.0
# Objective: sum_{i=1}^{N-1} [(x[i]^2 + x[i+1]^2)^2 + (-4*x[i] + 3)]
#   E(i) groups: L2 group type (no scale), elements SQ(x[i]) + SQ(x[i+1])
#     contribution = (x[i]^2 + x[i+1]^2)^2
#   L(i) groups: identity type, linear -4*x[i], constant RHS = -3
#     v = -4*x[i] - (-3) = -4*x[i] + 3, contribution = -4*x[i] + 3
# Source: Ph.L. Toint (1983), problem 31; Buckley#172
function engval1_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(T(-Inf), N), uvar = fill(T(Inf), N),
                      start = fill(T(2.0), N))
    # E(i) groups: (x[i]^2 + x[i+1]^2)^2  for i = 1..N-1
    ExaModels.@add_obj(c, (x[i]^2 + x[i+1]^2)^2 for i in 1:N-1)
    # L(i) groups: -4*x[i] + 3  for i = 1..N-1
    ExaModels.@add_obj(c, -T(4) * x[i] + T(3) for i in 1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
