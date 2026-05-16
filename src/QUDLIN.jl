# QUDLIN: Simple problem
# N vars, M=N/2 (default N=5000, M=2500)
# Bounds: 0 <= x_i <= 10 (default lvar=0, uvar=10)
# Objective (N type): sum_i C_i * x_i + 2PR elements
#   C_i = -10*i (SIF: RM C RI -10.0)
#   2PR elements E(I): F = X*Y = X(I)*X(I+1) for I=1..M
# OBJ has linear part and 2PR elements:
#   sum_i (-10*i)*x_i + sum_{I=1}^{M} x_I * x_{I+1}
function qudlin_model(N = 5000, M = 2500; T = Float64, backend = nothing, kwargs...)
    lvar = zeros(T, N)
    uvar = fill(T(10.0), N)
    start = zeros(T, N)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # Linear objective: sum_i (-10*i)*x_i
    ExaModels.@add_obj(c, T(-10) * T(i) * x[i] for i in 1:N)

    # 2PR elements: sum_{I=1}^{M} x_I * x_{I+1}
    ExaModels.@add_obj(c, x[i] * x[i+1] for i in 1:M)

    return ExaModels.ExaModel(c; kwargs...)
end
