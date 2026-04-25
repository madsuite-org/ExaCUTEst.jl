# EDENSCH: Extended Dennis-Schnabel problem (N variables, unconstrained)
# N = 2000 (SIF default), all variables free, start = 8.0
#
# Groups for i = 1..N-1:
#   A(i) type L4: v = x[i] - 2        → (x[i]-2)^4
#   B(i) type L2: v = -2*x[i+1] + x[i]*x[i+1]
#                   = x[i+1]*(x[i]-2) → (x[i+1]*(x[i]-2))^2
#   C(i) type L2: v = x[i+1] + 1      → (x[i+1]+1)^2
# Group A(N) type L4: v = 0 - 2 = -2  → (-2)^4 = 16 (constant term)
#
# Objective:
#   sum_{i=1}^{N-1} [ (x[i]-2)^4 + (x[i+1]*(x[i]-2))^2 + (x[i+1]+1)^2 ] + 16
function edensch_model(N = 2000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(8.0))
    # Constant term from A(N): v = 0*x[N] - 2 = -2, L4 → (-2)^4 = 16
    # Use x[N]*0 + 16 pattern to give ExaModels a variable reference
    ExaModels.@add_obj(c, x[N] * T(0) + T(16))
    # Loop over i = 1..N-1
    ExaModels.@add_obj(c, (x[i] - T(2))^4 + (x[i+1] * (x[i] - T(2)))^2 + (x[i+1] + T(1))^2 for i in 1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
