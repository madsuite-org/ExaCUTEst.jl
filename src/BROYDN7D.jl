# BROYDN7D: Seven-diagonal Broyden problem (N variables, unconstrained)
# Default N/2 = 2500, so N = 5000 variables, all free, start = 1.0
# GROUP TYPE L7/3: F(v) = |v|^(7/3)
# ELEMENT BROY(V1): F(V1) = (3 - 2*V1) * V1
#   (TEMP1 = -2*V1, TEMP2 = 3 + TEMP1 = 3 - 2*V1, F = TEMP2 * V1)
#
# E(1):   v = BROY(X(1)) - 2*X(2) - (-1) = (3-2*x1)*x1 - 2*x2 + 1
# E(I) for I=2..N-1: v = BROY(X(I)) - X(I-1) - 2*X(I+1) + 1
# E(N):   v = BROY(X(N)) - X(N-1) + 1
# S(I) for I=1..N/2: v = X(I) + X(I + N/2)
# All groups use L7/3: contribution = |v|^(7/3)
function broydn7d_model(N2 = 2500; T = Float64, backend = nothing, kwargs...)
    N = 2 * N2
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))
    # E(1): v = (3-2*x1)*x1 - 2*x2 + 1
    ExaModels.@add_obj(c, abs((3 - 2 * x[1]) * x[1] - 2 * x[2] + 1)^(T(7 / 3)))
    # E(I) for I=2..N-1: v = (3-2*x_i)*x_i - x_{i-1} - 2*x_{i+1} + 1
    ExaModels.@add_obj(
        c,
        abs((3 - 2 * x[i]) * x[i] - x[i-1] - 2 * x[i+1] + 1)^(T(7 / 3)) for i in 2:N-1
    )
    # E(N): v = (3-2*x_N)*x_N - x_{N-1} + 1
    ExaModels.@add_obj(c, abs((3 - 2 * x[N]) * x[N] - x[N-1] + 1)^(T(7 / 3)))
    # S(I) for I=1..N/2: v = x_i + x_{i+N/2}
    ExaModels.@add_obj(c, abs(x[i] + x[i+N2])^(T(7 / 3)) for i in 1:N2)
    return ExaModels.ExaModel(c; kwargs...)
end
