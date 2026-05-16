# CYCLIC3: Cyclic cubic system (N+2 variables, N constraints)
# Default N = 100000
# Variables: X(1)..X(N+2), all free, start = 1000.0
# Constraints E(I) for I=1..N:   X(I)^3 - X(I+1)*X(I+2) = 0
# E(N+1): X(N+1) - X(1) = 0  (linear)
# E(N+2): X(N+2) - X(2) = 0  (linear)
function cyclic3_model(N = 100000; T = Float64, backend = nothing, kwargs...)
    NV = N + 2
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = T(1000))
    # Linear groups first (CUTEst ordering): E(N+1) and E(N+2) before nonlinear E(1..N)
    ExaModels.@add_con(c, x[N+1] - x[1])
    ExaModels.@add_con(c, x[N+2] - x[2])
    # E(I) for I=1..N: X(I)^3 - X(I+1)*X(I+2) = 0
    ExaModels.@add_con(c, x[i]^3 - x[i+1] * x[i+2] for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
