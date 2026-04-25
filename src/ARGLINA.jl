# ARGLINA: Variable dimension full rank linear problem (N=200 variables, M=400 groups)
# minimize sum_{i=1}^{M} v_i^2   (L2 group type, no scale → contribution = v^2)
# For i=1..N:   v_i = x_i + a*sum_j x_j - 1  where a = -2/M
# For i=N+1..M: v_i = a*sum_j x_j - 1
#
# Algebraic simplification (using a = -2/M, so 2a + M*a^2 = -4/M + 4/M = 0):
#   f(x) = sum_i x_i^2 + 2*sum_i x_i + M
#
# BOUNDS: FR DEFAULT → all variables free; START: DEFAULT = 1.0
# Source: More', Garbow, Hillstrom (1981), problem 32
function arglina_model(N = 200, M = 400; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf),
                                 uvar = T(Inf),
                                 start = T(1.0))

    # f(x) = sum_i (x_i^2 + 2*x_i) + M
    # Include constant M in the first term to avoid unsupported scalar @add_obj
    ExaModels.@add_obj(c, x[1]^2 + T(2) * x[1] + T(M))
    ExaModels.@add_obj(c, x[i]^2 + T(2) * x[i] for i in 2:N)

    return ExaModels.ExaModel(c; kwargs...)
end
