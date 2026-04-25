# ARWHDNE: Arrow-head nonlinear equations problem (N variables, 2*(N-1) constraints)
# Zero objective; constraints:
#   L(I) for I=1..N-1: -4*X(I) - (-3) = 3 - 4*X(I) = 0
#   G(I) for I=1..N-1: X(I)^2 + X(N)^2 = 0
# All variables free, starting point x = 1.0
# Default N = 500 (matching SIF default)
#
# SIF structure:
#   NGS = N - 1
#   L(I) groups (E type, identity): linear -4*X(I), constant -3
#     → v = -4*X(I) - (-3) = 3 - 4*X(I)
#   G(I) groups (E type, identity): elements A(I)=X(I)^2, B(I)=X(N)^2
#     → v = X(I)^2 + X(N)^2
#   Constraint ordering: L(I) purely linear → come before G(I) in CUTEst
function arwhdne_model(N = 500; T = Float64, backend = nothing, kwargs...)
    NGS = N - 1
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))
    # Zero objective (NOR2 problem)
    # L(I) constraints: 3 - 4*x[i] = 0  (purely linear, appear first in CUTEst)
    ExaModels.@add_con(c, T(3) - T(4) * x[i] for i in 1:NGS)
    # G(I) constraints: x[i]^2 + x[N]^2 = 0
    ExaModels.@add_con(c, x[i]^2 + x[N]^2 for i in 1:NGS)
    return ExaModels.ExaModel(c; kwargs...)
end
