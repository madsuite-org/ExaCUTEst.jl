# DIXMAANE: Dixon-Maany test problem (version E), unconstrained
# M = 1000 (default), N = 3*M = 3000 variables, all free, start = 2.0
#
# Objective (4 groups, all N type, identity group function):
#   GA = sum_{i=1}^{N}   AI * x_i^2  +  1   where AI = ALPHA*(I/N)^K1 = I/N
#                                             (constant from CONSTANTS: RHS=-1 → +1)
#   GB: BETA=0.0, K2=0 → all coefficients are 0 — omitted
#   GC = sum_{i=1}^{2M}  GAMMA * x_i^2 * x_{i+M}^4          (GAMMA=0.125, K3=0)
#   GD = sum_{i=1}^{M}   DI * x_i * x_{i+2M}  where DI = DELTA*(I/N)^K4 = 0.125*I/N
#
# Parameters: ALPHA=1.0, BETA=0.0, GAMMA=0.125, DELTA=0.125, K1=1, K2=0, K3=0, K4=1

function dixmaane_model(M = 1000; T = Float64, backend = nothing, kwargs...)
    N = 3 * M

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(2))

    # GA: sum_{i=1}^{N} AI * x_i^2 + 1.0  where AI = I/N  (K1=1, ALPHA=1.0)
    # constant +1 from RHS=-1
    ExaModels.@add_par(c, AN, T[i / N for i in 1:N])
    ExaModels.@add_obj(c, AN[i] * x[i]^2 for i in 1:N)
    ExaModels.@add_obj(c, T(1) for i in 1:1)

    # GB: BETA=0.0 → contributes nothing — omitted

    # GC: sum_{i=1}^{2M} GAMMA * x_i^2 * x_{i+M}^4   (GAMMA=0.125, K3=0)
    ExaModels.@add_obj(c, T(0.125) * x[i]^2 * x[i+M]^4 for i in 1:2M)

    # GD: sum_{i=1}^{M} DI * x_i * x_{i+2M}  where DI = 0.125*I/N  (K4=1, DELTA=0.125)
    ExaModels.@add_par(c, DN, T[0.125 * i / N for i in 1:M])
    ExaModels.@add_obj(c, DN[i] * x[i] * x[i+2M] for i in 1:M)

    return ExaModels.ExaModel(c; kwargs...)
end
