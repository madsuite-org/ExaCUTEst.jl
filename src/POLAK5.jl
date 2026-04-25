# POLAK5: Nonlinear minmax in two variables (Polak, Mayne & Higgins 1991)
# 3 free variables: X1, X2, U; start = [0.1, 0.1, 0]
# Objective: U; 2 L-type constraints
# EL(S) = (X1 - X2⁴ + S)²,  SQ = X1²
# F1: -U + 3*X1² + 50*(X1 - X2⁴ - 1)² ≤ 0
# F2: -U + 3*X1² + 50*(X1 - X2⁴ + 1)² ≤ 0
function polak5_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(0.1), T(0.1), T(0)])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c,
        -x[3] + T(3)*x[1]^2 + T(50)*(x[1] - x[2]^4 - T(1))^2;
        lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c,
        -x[3] + T(3)*x[1]^2 + T(50)*(x[1] - x[2]^4 + T(1))^2;
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
