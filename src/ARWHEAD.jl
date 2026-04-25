# ARWHEAD: Arrow-head quartic problem (N variables, unconstrained)
# minimize sum_{i=1}^{N-1} [(-4*x_i + 3) + (x_i^2 + x_N^2)^2]
# All variables free, starting point x = 1.0
# Default N = 5000 (matching SIF default IE N = 5000)
#
# SIF structure:
#   NGS = N - 1
#   L(I) groups (identity type): v = -4*X(I) - (-3.0) = -4*X(I) + 3
#   G(I) groups (L2 type, no scale): v = A(I) + B(I) = X(I)^2 + X(N)^2
#     → contribution = v^2 = (X(I)^2 + X(N)^2)^2
function arwhead_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    NGS = N - 1
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))
    # L(I) groups: identity type, contribution = -4*x_i + 3
    ExaModels.@add_obj(c, (-4 * x[i] + 3) for i in 1:NGS)
    # G(I) groups: L2 type, contribution = (x_i^2 + x_N^2)^2
    ExaModels.@add_obj(c, (x[i]^2 + x[N]^2)^2 for i in 1:NGS)
    return ExaModels.ExaModel(c; kwargs...)
end
