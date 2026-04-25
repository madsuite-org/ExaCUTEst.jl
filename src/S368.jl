# S368: Wolfe's problem (N=8 variables, unconstrained)
# minimize sum_{I,J=1}^{N} (-x[I]^2*x[J]^4 + x[I]^3*x[J]^3)
# Source: P. Wolfe (1972), see also Schittkowski #368
# SIF: N^2 groups M(I,J) with element PRODM = -x[I]^2*x[J]^4,
#      N^2 groups P(I,J) with element PRODP = x[I]^3*x[J]^3;
#      all identity (no GROUP TYPE), coefficient 1.0
# Variable bounds: [0, 1] (SIF 'UP DEFAULT' 1.0; lower default = 0)
# Start: x[I] = I/(N+1) (SIF RI arithmetic)
# Default N = 8 (matching Schittkowski #368)
function s368_model(N = 8; T = Float64, backend = nothing, kwargs...)
    start_vec = T[i / (N + 1) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0.0), uvar = T(1.0), start = start_vec)
    ExaModels.@add_obj(c, -x[i]^2 * x[j]^4 + x[i]^3 * x[j]^3 for i in 1:N, j in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
