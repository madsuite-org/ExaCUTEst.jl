# BIGGSB1: Bounded quadratic test problem
# Source: Bartholomew-Biggs and Hernandez, 1992; SIF: Ph. Toint, April 1992
#
# N=5000 variables; GROUP TYPE L2: F(v) = v^2
#   lvar=0, uvar=0.9 for x[1..N-1]
#   lvar=-Inf, uvar=Inf for x[N] (XR bound)
# Start: 0 (default)
#
# G(0): v = X(1) - 1   → (x[1]-1)^2
# G(I) for I=1..N-1: v = X(I+1) - X(I)  → (x[i+1]-x[i])^2
# G(N): v = -X(N) - (-1) = 1 - X(N)  → (1-x[N])^2
function biggsb1_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    lvar = [fill(T(0), N - 1); T(-Inf)]
    uvar = [fill(T(0.9), N - 1); T(Inf)]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = T(0))
    ExaModels.@add_obj(c, (x[1] - T(1))^2)
    ExaModels.@add_obj(c, (x[i + 1] - x[i])^2 for i in 1:(N - 1))
    ExaModels.@add_obj(c, (T(1) - x[N])^2)
    return ExaModels.ExaModel(c; kwargs...)
end
