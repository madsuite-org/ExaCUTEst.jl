# POLAK1: Nonlinear minmax in two variables (Polak, Mayne & Higgins 1991)
# 3 free variables: X1, X2, U; start = [50, 0.05, 0]
# Objective: U; 2 L-type constraints
# F1: -U + exp(0.001*X1² + (X2-1)²) ≤ 0
# F2: -U + exp(0.001*X1² + (X2+1)²) ≤ 0
function polak1_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(50), T(0.05), T(0)])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c,
        -x[3] + exp(T(0.001)*x[1]^2 + (x[2] - T(1))^2);
        lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c,
        -x[3] + exp(T(0.001)*x[1]^2 + (x[2] + T(1))^2);
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
