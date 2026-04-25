# ARGLINB: Variable dimension rank-one linear problem (N=200 vars, M=400 groups)
# Source: More', Garbow, Hillstrom (1981), problem 33
# Classification: SUR2-AN-V-0 (sum of squares, unconstrained)
#
# SIF: group G(I) for I=1..M has type L2 (φ(v)=v²), rhs=1
#   v_I = sum_{j=1}^N I*j*X(j) - 1 = I * (sum_j j*X(j)) - 1 = I*S - 1
#   where S = sum_{j=1}^N j*X(j)
#
# f = sum_{I=1}^M (I*S - 1)^2
#   = C1*S^2 - 2*C2*S + M
#   where C1 = sum_{I=1}^M I^2 = M*(M+1)*(2M+1)/6
#         C2 = sum_{I=1}^M I   = M*(M+1)/2
#
# Expanding S^2 = sum_j sum_k j*k*x[j]*x[k]:
#   f = C1*sum_j sum_k j*k*x[j]*x[k] - 2*C2*sum_j j*x[j] + M
#
# All free; start: X = 1.0
function arglinb_model(N = 200, M = 400; T = Float64, backend = nothing, kwargs...)
    C1 = T(M) * T(M + 1) * T(2M + 1) / T(6)
    C2 = T(M) * T(M + 1) / T(2)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1))
    # C1 * sum_j sum_k j*k*x[j]*x[k]  (dense 2D quadratic)
    # Loop variables j, k used directly as numeric multipliers (not wrapped in T())
    ExaModels.@add_obj(c, C1 * j * k * x[j] * x[k] for j in 1:N, k in 1:N)
    # -2*C2 * sum_j j*x[j]  (linear)
    ExaModels.@add_obj(c, T(-2) * C2 * j * x[j] for j in 1:N)
    # constant M
    ExaModels.@add_obj(c, T(M) for _ in 1:1)
    return ExaModels.ExaModel(c; kwargs...)
end
