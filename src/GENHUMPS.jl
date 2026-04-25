# GENHUMPS: generalized humps problem (N = 5000 variables, unconstrained)
# minimize sum_{i=1}^{N-1} [(sin(ZETA*x[i])*sin(ZETA*x[i+1]))^2 + 0.05*x[i]^2 + 0.05*x[i+1]^2]
# All variables free (FR DEFAULT)
# Start: x[1] = -506.0, x[2..N] = -506.2
# ZETA = 20.0 (active RE line)
# Default N = 5000
# Source: Ph. Toint, private communication, 1997; SIF by N. Gould and Ph. Toint, November 1997
#
# SIF structure:
#   Single group OBJ (N type, identity function)
#   Element HMP(X,Y;A): F = (sin(A*X)*sin(A*Y))^2, A=ZETA=20.0, coeff 1.0
#   Element SQ(X): F = X^2, coeffs 0.05 for SX(i) and SY(i)
#   Group value = sum of all element contributions
function genhumps_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    ZETA = T(20.0)
    start = T[-506.2 for _ in 1:N]
    start[1] = T(-506.0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    # HMP elements: (sin(ZETA*x[i]) * sin(ZETA*x[i+1]))^2, coeff 1.0
    ExaModels.@add_obj(c, (sin(ZETA * x[i]) * sin(ZETA * x[i+1]))^2 for i in 1:N-1)
    # SQ elements with coeff 0.05: SX(i) = 0.05*x[i]^2, SY(i) = 0.05*x[i+1]^2
    ExaModels.@add_obj(c, T(0.05) * x[i]^2 for i in 1:N-1)
    ExaModels.@add_obj(c, T(0.05) * x[i+1]^2 for i in 1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
