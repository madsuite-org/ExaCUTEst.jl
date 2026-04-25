# JENSMP: Jennrich and Sampson problem (2 variables, unconstrained)
# minimize sum_{i=1}^{10} (exp(i*x1) + exp(i*x2) - (2+2*i))^2
# Source: More', Garbow, Hillstrom (1981), problem 6
# SIF: 10 L2 groups, no scale; PEXP element: F = exp(P*V);
#   CONSTANT: G(I) = 2+2*I;  BOUNDS: FR DEFAULT (free variables)
function jensmp_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[0.3, 0.4])
    # Group G(I): v = exp(I*x1) + exp(I*x2) - (2+2*I); contribution = v^2
    ExaModels.@add_obj(c, (exp(i * x[1]) + exp(i * x[2]) - (2 + 2*i))^2 for i in 1:10)
    return ExaModels.ExaModel(c; kwargs...)
end
