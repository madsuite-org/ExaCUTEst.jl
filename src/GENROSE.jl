# GENROSE: Generalized Rosenbrock function (N=500 variables, unconstrained)
# minimize 1 + sum_{i=2}^{N} [100*(x[i] - x[i-1]^2)^2 + (x[i] - 1)^2]
# All variables free; start: x[i] = i/(N+1)
#
# SIF structure:
#   OBJ group: L2 type, constant RHS=-1, value = 0 - (-1) = 1, contribution = 1^2 = 1
#   Q(I) groups (I=2..N): L2 type, SCALE=0.01
#     linear 1.0*X(I) + element MSQR(X(I-1)) = -X(I-1)^2
#     value = X(I) - X(I-1)^2
#     contribution = (X(I) - X(I-1)^2)^2 / 0.01 = 100*(X(I) - X(I-1)^2)^2
#   L(I) groups (I=2..N): L2 type, linear 1.0*X(I), constant RHS=1.0
#     value = X(I) - 1
#     contribution = (X(I) - 1)^2
function genrose_model(N = 500; T = Float64, backend = nothing, kwargs...)
    x0 = T[i / (N + 1) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)
    # OBJ group: L2 type, constant RHS=-1, value=1, contribution=1^2=1
    # expressed as x[1]*0 + 1 to give ExaModels a variable reference
    ExaModels.@add_obj(c, x[1] * T(0) + T(1))
    # Q(I) groups: 100*(x[i] - x[i-1]^2)^2
    ExaModels.@add_obj(c, T(100) * (x[i] - x[i-1]^2)^2 for i in 2:N)
    # L(I) groups: (x[i] - 1)^2
    ExaModels.@add_obj(c, (x[i] - T(1))^2 for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end
