# NONDIA: Shanno nondiagonal extension of Rosenbrock, N=5000 variables, unconstrained
# minimize (x[1]-1)^2 + sum_{i=2}^{N} 100*(x[1] - x[i-1]^2)^2
# All variables free; start: x[i] = -1.0
#
# SIF structure:
#   SQ(1) group:  L2 type, no scale
#     linear part X(1), constant = 1.0 → v = x[1] - 1
#     contribution = (x[1] - 1)^2
#   SQ(I) groups for I=2..N:  L2 type, SCALE = 0.01
#     linear part X(1) (first variable, not X(I)), element ELA(I) = -X(I-1)^GAMMA = -X(I-1)^2
#     v = x[1] + (-x[I-1]^2) = x[1] - x[I-1]^2
#     contribution = v^2 / 0.01 = 100*(x[1] - x[I-1]^2)^2
function nondia_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(-1.0))
    ExaModels.@add_obj(c, (x[1] - T(1))^2)
    ExaModels.@add_obj(c, T(100) * (x[1] - x[i-1]^2)^2 for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end
