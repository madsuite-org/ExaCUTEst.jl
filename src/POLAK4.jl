# POLAK4: Nonlinear minmax in two variables (Polak, Mayne & Higgins 1991)
# 3 free variables: X1, X2, U; start = [0.9, 0.1, 0]
# Objective: U; 3 L-type constraints
# SSQ(XX,S) = (XX - S)²
# F1: -U - X1 + 2*X1² + 2*X2²        - 1 ≤ 0
# F2: -U       + 0.01*X1² + 0.01*X2² - 0.01 ≤ 0
# F3: -U       + 100000*(X1-2)² + X2² - 100000 ≤ 0
function polak4_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(0.9), T(0.1), T(0)])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c,
        -x[3] - x[1] + T(2)*x[1]^2 + T(2)*x[2]^2 - T(1);
        lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c,
        -x[3] + T(0.01)*x[1]^2 + T(0.01)*x[2]^2 - T(0.01);
        lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c,
        -x[3] + T(100000)*(x[1] - T(2))^2 + x[2]^2 - T(100000);
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
