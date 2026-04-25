# DIXMAANB: Dixon-Maany test problem (version B), unconstrained
# M = 1000 (default), N = 3*M = 3000 variables, all free, start = 2.0
#
# Objective (4 groups, all N type, identity group function):
#   GA = sum_{i=1}^{N}   ALPHA * x_i^2  +  1   (constant from CONSTANTS: RHS=-1 → +1)
#   GB = sum_{i=1}^{N-1} BETA  * x_i^2 * (x_{i+1} + x_{i+1}^2)^2   (BETA=0.0625)
#   GC = sum_{i=1}^{2M}  GAMMA * x_i^2 * x_{i+M}^4                  (GAMMA=0.0625)
#   GD = sum_{i=1}^{M}   DELTA * x_i   * x_{i+2M}                   (DELTA=0.0625)
#
# Parameters: ALPHA=1.0, BETA=0.0625, GAMMA=0.0625, DELTA=0.0625, K1=K2=K3=K4=0
# (all K=0 so coefficients are just the base constants)

function dixmaanb_model(M = 1000; T = Float64, backend = nothing, kwargs...)
    N = 3 * M

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(2))

    # GA: sum_{i=1}^{N} ALPHA * x_i^2 + 1.0   (ALPHA=1.0, constant from RHS=-1 → +1)
    # constant +1 folded into the i=1 term; rest of the sum handled separately
    ExaModels.@add_obj(c, x[i]^2 + T(1) for i in 1:1)
    ExaModels.@add_obj(c, x[i]^2 for i in 2:N)

    # GB: sum_{i=1}^{N-1} BETA * x_i^2 * (x_{i+1} + x_{i+1}^2)^2   (BETA=0.0625)
    ExaModels.@add_obj(c, T(0.0625) * x[i]^2 * (x[i+1] + x[i+1]^2)^2 for i in 1:N-1)

    # GC: sum_{i=1}^{2M} GAMMA * x_i^2 * x_{i+M}^4   (GAMMA=0.0625)
    ExaModels.@add_obj(c, T(0.0625) * x[i]^2 * x[i+M]^4 for i in 1:2M)

    # GD: sum_{i=1}^{M} DELTA * x_i * x_{i+2M}   (DELTA=0.0625)
    ExaModels.@add_obj(c, T(0.0625) * x[i] * x[i+2M] for i in 1:M)

    return ExaModels.ExaModel(c; kwargs...)
end
