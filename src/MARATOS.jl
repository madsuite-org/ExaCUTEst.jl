# MARATOS: Maratos problem with penalty parameter 1e-6 (Brown & Bartholomew-Biggs 1987)
# 2 free variables: X1, X2; start = [1.1, 0.1]
# Objective: -X1 + 1e-6*(X1^2 + X2^2 - 1)
# 1 equality constraint: X1^2 + X2^2 - 1 = 0
function maratos_model(; T = Float64, backend = nothing, kwargs...)
    TAU = T(1) / T(1000000)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(1.1), T(0.1)])
    ExaModels.@add_obj(c, -x[1] + TAU * (x[1]^2 + x[2]^2 - T(1)))
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
