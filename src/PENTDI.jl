# PENTDI: 5-diagonal convex QP
# Variables X(1..N) >= 0 (default bounds)
# Objective:
#   OBJ0: 6 * sum_i X(I)^2
#   OBJ1: linear in X(1), X(N/2-1), X(N/2), X(N/2+1) with specific coefficients
#          and X(I) for I=N/2+3..N with coefficient 1.0
#   OBJ: -4*sum_{i=N+1..P, step 2} Z(I) + 1*sum_{i=N+1..P, step 2} Z(I+1)
#         where elements are:
#         Z(I) = 2PR(X(idx1), X(idx2)) grouped with coeff -4.0
#         Z(I+1) = 2PR(X(idx1), X(idx3)) grouped with coeff 1.0
# Interpretation: OBJ includes cross terms:
#   -4 * X(I)*X(I+1) + 1 * X(I)*X(I+2) for I=1..N-2
# 2PR element: F = X*Y
# OBJ (unconstrained, no bounds section)
#
# Let me re-analyze: GROUPS section has:
#   XN OBJ (no vars -- will receive element contributions only?)
#   XN OBJ0 (no vars)
#   XN OBJ1 X(1) -3.0  X(2) 1.0  X(N/2-1) 1.0  X(N/2) -3.0  X(N/2+1) 4.0
#            and X(I) 1.0 for I=N/2+3..N
# GROUP USES:
#   OBJ0: Z(I) with coeff 6.0 for I=1..N (SQ elements)
#   OBJ: Z(I) with coeff -4.0, Z(I+1) with coeff 1.0 for 2PR elements (I=N+1..P step 2)
#
# So objective = OBJ + OBJ0 + OBJ1
#   OBJ = sum_{I=1..N-2} (-4*X(I)*X(I+1) + 1*X(I)*X(I+2))
#   OBJ0 = 6 * sum_i X(I)^2
#   OBJ1 = -3*X(1) + 1*X(2) + ... (linear terms)
function pentdi_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    # Default bounds: x >= 0 (SIF default)
    lvar = zeros(T, N)
    uvar = fill(T(Inf), N)
    start = zeros(T, N)

    N2 = N ÷ 2

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # OBJ0: 6*sum X(I)^2
    ExaModels.@add_obj(c, T(6) * x[i]^2 for i in 1:N)

    # OBJ1: linear terms
    # X(1) -3, X(2) 1, X(N/2-1) 1, X(N/2) -3, X(N/2+1) 4
    # X(I) 1 for I=N/2+3..N
    ExaModels.@add_obj(c, T(-3) * x[1] + T(1) * x[2])
    if N2 - 1 >= 3  # avoid double-counting
        ExaModels.@add_obj(c, T(1) * x[N2-1])
    end
    ExaModels.@add_obj(c, T(-3) * x[N2] + T(4) * x[N2+1])
    ExaModels.@add_obj(c, x[i] for i in N2+3:N)

    # OBJ: -4*X(I)*X(I+1) + X(I)*X(I+2) for I=1..N-2
    ExaModels.@add_obj(c, T(-4) * x[i] * x[i+1] + x[i] * x[i+2] for i in 1:N-2)

    return ExaModels.ExaModel(c; kwargs...)
end
