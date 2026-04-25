# CHNRSNBM: Chained Rosenbrock (modified, Luksan et al.), N=50 variables, unconstrained
# minimize sum_{i=2}^{N} [ 16*ALPHA(i)^2 * (x[i-1] - x[i]^2)^2 + (x[i] - 1)^2 ]
# where ALPHA(i) = 1.5 + sin(i)  (i is the integer loop index)
#
# SIF structure:
#   ALPHA_I = 1.5 + sin(RI) where RI = real(I)
#   SCL = 1 / (16 * ALPHA_I^2)
#   Groups SQ(I): linear X(I-1), element ELA(I) = -X(I)^2 → v = X(I-1) - X(I)^2
#                 L2 group type, scale σ = SCL
#                 contribution = v^2 / σ = 16*ALPHA_I^2 * (X(I-1) - X(I)^2)^2
#   Groups B(I):  linear X(I), constant 1 → v = X(I) - 1
#                 L2 group type, no scale → contribution = (X(I) - 1)^2
#   Bounds: FR DEFAULT → all variables free
#   Start: all -1.0
function chnrsnbm_model(N = 50; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(-1.0))
    # Precompute scale coefficients: 16 * ALPHA(i)^2 where ALPHA(i) = 1.5 + sin(i)
    # indexed for i = 2..N → array index 1..N-1
    ExaModels.@add_par(c, A, T[16 * (1.5 + sin(i))^2 for i in 2:N])
    # SQ(I) groups: 16*ALPHA(I)^2 * (x[I-1] - x[I]^2)^2
    ExaModels.@add_obj(c, A[i-1] * (x[i-1] - x[i]^2)^2 for i in 2:N)
    # B(I) groups: (x[I] - 1)^2
    ExaModels.@add_obj(c, (x[i] - T(1))^2 for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end
