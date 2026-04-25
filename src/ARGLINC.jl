# ARGLINC: Variable dimension rank-one linear problem with zero rows/columns (N=200, M=400)
# Source: More', Garbow, Hillstrom (1981), problem 34
# Classification: SUR2-AN-V-0 (sum of squares, unconstrained)
#
# SIF: G(1) and G(M) are empty (zero contribution). G(I) for I=2..M-1 has type L2, rhs=1:
#   v_I = sum_{j=2}^{N-1} (I-1)*j*X(j) - 1 = (I-1)*S2 - 1
#   where S2 = sum_{j=2}^{N-1} j*X(j)   (excludes x[1] and x[N])
#
# f = sum_{I=2}^{M-1} ((I-1)*S2 - 1)^2
# Substituting k = I-1 (k from 1 to M-2):
#   f = sum_{k=1}^{M-2} (k*S2 - 1)^2
#     = C1'*S2^2 - 2*C2'*S2 + (M-2)
#   where C1' = sum_{k=1}^{M-2} k^2 = (M-2)*(M-1)*(2M-3)/6
#         C2' = sum_{k=1}^{M-2} k   = (M-2)*(M-1)/2
#
# Expanding S2^2 = sum_{j=2}^{N-1} sum_{k=2}^{N-1} j*k*x[j]*x[k]:
#   f = C1'*sum_j sum_k j*k*x[j]*x[k] - 2*C2'*sum_j j*x[j] + (M-2)
#   (both sums over j,k from 2 to N-1)
#
# All free; start: X = 1.0
function arglinc_model(N = 200, M = 400; T = Float64, backend = nothing, kwargs...)
    Mp = M - 2  # number of active groups: I=2..M-1, i.e. M-2 groups
    C1p = T(Mp) * T(Mp + 1) * T(2Mp + 1) / T(6)
    C2p = T(Mp) * T(Mp + 1) / T(2)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1))
    # C1' * sum_{j=2}^{N-1} sum_{k=2}^{N-1} j*k*x[j]*x[k]  (dense 2D quadratic over inner vars)
    # Loop variables j, k used directly as numeric multipliers (not wrapped in T())
    ExaModels.@add_obj(c, C1p * j * k * x[j] * x[k] for j in 2:N-1, k in 2:N-1)
    # -2*C2' * sum_{j=2}^{N-1} j*x[j]  (linear, inner variables only)
    ExaModels.@add_obj(c, T(-2) * C2p * j * x[j] for j in 2:N-1)
    # constant (M-2)
    ExaModels.@add_obj(c, T(Mp) for _ in 1:1)
    return ExaModels.ExaModel(c; kwargs...)
end
