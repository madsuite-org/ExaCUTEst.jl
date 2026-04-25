# EG1: LANCELOT manual example 1.2.3 (Conn et al. 1992)
# 3 variables: X1 (free), X2 ∈ [-1,1], X3 ∈ [1,2]; start = [0, 0, 1]
# Objective = X1^2 + (X2*X3)^4 + X2 + X2*sin(X1+X3) + X1*X3
# No constraints
# GROUP1 (GTYPE1: F=α²): α = X1  → X1²
# GROUP2 (GTYPE2: F=α⁴): α = X2*X3  → (X2*X3)⁴
# GROUP3 (identity): linear X2 + elements X2*sin(X1+X3) + X1*X3
function eg1_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[T(-Inf), T(-1), T(1)],
        uvar = T[T(Inf), T(1), T(2)],
        start = T[T(0), T(0), T(1)])
    ExaModels.@add_obj(c, x[1]^2)
    ExaModels.@add_obj(c, (x[2] * x[3])^4)
    ExaModels.@add_obj(c, x[2] + x[2] * sin(x[1] + x[3]) + x[1] * x[3])
    return ExaModels.ExaModel(c; kwargs...)
end
