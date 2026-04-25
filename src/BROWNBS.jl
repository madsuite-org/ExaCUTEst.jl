# BROWNBS: Brown badly scaled problem (2 variables, unconstrained)
# minimize (x1 - 1e6)^2 + (x2 - 2e-6)^2 + (x1*x2 - 2)^2
# Source: More', Garbow, Hillstrom (1981), problem 4; Buckley#25
# SIF: N=2 variables, all free (FR 'DEFAULT'), start=1.0
#   For I=1 (N-1=1 group sets):
#     A(1): linear x[1]*1.0, RHS=1e6, L2 → (x[1] - 1e6)^2
#     B(1): linear x[2]*1.0, RHS=2e-6, L2 → (x[2] - 2e-6)^2
#     C(1): element 2PR(x[1],x[2])=x[1]*x[2], RHS=2.0, L2 → (x[1]*x[2] - 2)^2
function brownbs_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))
    ExaModels.@add_obj(c, (x[1] - T(1000000))^2 + (x[2] - T(0.000002))^2 + (x[1] * x[2] - T(2))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
