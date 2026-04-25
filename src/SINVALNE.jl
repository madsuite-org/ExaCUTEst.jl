# SINVALNE: Sine variant of Rosenbrock equations (Xiao & Zhou 1992)
# 2 free variables: X1, X2; start = [4.712389, -1.0] ≈ [3π/2, -1]
# No objective; 2 equality constraints (scale ignored for identity groups):
#   G1: X2 - sin(X1) = 0
#   G2: X1 = 0
function sinvalne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(4.712389), T(-1.0)])
    ExaModels.@add_con(c, x[2] - sin(x[1]))
    ExaModels.@add_con(c, x[1])
    return ExaModels.ExaModel(c; kwargs...)
end
