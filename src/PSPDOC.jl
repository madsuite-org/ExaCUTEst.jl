# PSPDOC: Partially separable objective (Toint 1983)
# 4 variables: X1 ∈ (-∞,-1], X2..X4 free; start = 3.0
# Objective: sum of sqrt group types
# GROUP TYPE SQR: contribution = sqrt(v) where v = element_sum - rhs
# G(1): sqrt(X1² + (X2-X3)² + 1)
# G(2): sqrt(X2² + (X3-X4)² + 1)
# rhs = -1 for all groups (default constant -1), so v = elems - (-1) = elems + 1
function pspdoc_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
        lvar = T[T(-Inf), T(-Inf), T(-Inf), T(-Inf)],
        uvar = T[T(-1), T(Inf), T(Inf), T(Inf)],
        start = T(3))
    ExaModels.@add_obj(c, sqrt(x[1]^2 + (x[2] - x[3])^2 + T(1)))
    ExaModels.@add_obj(c, sqrt(x[2]^2 + (x[3] - x[4])^2 + T(1)))
    return ExaModels.ExaModel(c; kwargs...)
end
