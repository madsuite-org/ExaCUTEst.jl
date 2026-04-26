# CUBENE: Cubic variant of Rosenbrock (equations version)
# 2 free variables: X1, X2; start = [-1.2, 1.0]
# No objective; 2 equality constraints:
#   SQ(1): X1 - 1 = 0
#   SQ(2): SCALE=0.1, v = X2 - X1^3; CUTEst reports v/σ → (X2 - X1^3)/0.1
function cubene_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(-1.2), T(1.0)])
    ExaModels.@add_con(c, x[1] - T(1))
    ExaModels.@add_con(c, (x[2] - x[1]^3) / T(0.1))
    return ExaModels.ExaModel(c; kwargs...)
end
