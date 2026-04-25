# EXPLIN2: Problem with mixed exponential and linear terms (N=1200, M=100)
# minimize sum_{i=1}^{N} (-10*i)*x[i] + sum_{i=1}^{M} exp(0.1*(i/M)*x[i]*x[i+1])
# Variables: lvar=0, uvar=10, start=0
#
# SIF structure:
#   Linear part: RM C RI -10.0 → C = -10*I (coefficient of X(I))
#   Element E(I): EXP type with P=I/M, F = exp(0.1*P*X(I)*X(I+1))
#   OBJ group: identity type, sum of linear + element contributions
function explin2_model(N = 1200, M = 100; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0), uvar = T(10), start = T(0))
    # Linear part: sum_{i=1}^{N} (-10*i) * x[i]
    ExaModels.@add_par(c, θ, T[-10 * i for i in 1:N])
    ExaModels.@add_obj(c, θ[i] * x[i] for i in 1:N)
    # Exponential part: sum_{i=1}^{M} exp(0.1*(i/M)*x[i]*x[i+1])
    ExaModels.@add_par(c, P, T[i / M for i in 1:M])
    ExaModels.@add_obj(c, exp(T(0.1) * P[i] * x[i] * x[i+1]) for i in 1:M)
    return ExaModels.ExaModel(c; kwargs...)
end
