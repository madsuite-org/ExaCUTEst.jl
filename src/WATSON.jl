# WATSON: Watson problem in 12 variables (More', Garbow, Hillstrom 1981, problem 20)
# Nonlinear least squares: 31 L2 groups (no scale, F(v) = v^2)
#
# Variables: x[1]..x[12], all free (FR DEFAULT)
# Starting point: all zeros (no START section)
#
# t_i = i/29 for i = 1..29
# p(t)  = sum_{j=1}^{12} x[j] * t^(j-1)           (polynomial value)
# p'(t) = sum_{j=2}^{12} (j-1) * x[j] * t^(j-2)   (polynomial derivative)
#
# G(i), i=1..29 (L2, no scale):
#   v_i = p'(t_i) - p(t_i)^2 - 1
#   contribution = v_i^2
#
# G(30) (L2, no scale):
#   v_30 = x[1]   (XN G(30) X(1) 1.0; CONSTANT = 0.0 for G(30))
#   contribution = x[1]^2
#
# G(31) (L2, no scale):
#   v_31 = x[2] - x[1]^2 - 1   (XN G(M) X(2) 1.0; element E(M)=MSQ: -x[1]^2; CONSTANT=1)
#   contribution = (x[2] - x[1]^2 - 1)^2
#
# Precompute polynomial coefficients for each i=1..29:
#   A[12*(i-1)+j] = t_i^(j-1)         (j=1..12)
#   D[12*(i-1)+j] = (j-1)*t_i^(j-2)  (j=1..12; D for j=1 is 0)
function watson_model(; T = Float64, backend = nothing, kwargs...)
    N = 12  # number of variables

    # Precompute flattened coefficient arrays (length 29*12 = 348)
    # A[12*(i-1)+j] = t_i^(j-1)        polynomial basis coefficient
    # D[12*(i-1)+j] = (j-1)*t_i^(j-2)  derivative basis coefficient
    A = T[
        let i = div(k - 1, N) + 1, j = mod(k - 1, N) + 1, ti = i / 29.0
            j == 1 ? 1.0 : ti^(j - 1)
        end
        for k in 1:29*N
    ]
    D = T[
        let i = div(k - 1, N) + 1, j = mod(k - 1, N) + 1, ti = i / 29.0
            j == 1 ? 0.0 : (j - 1) * (j == 2 ? 1.0 : ti^(j - 2))
        end
        for k in 1:29*N
    ]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = zeros(T, N))
    ExaModels.@add_par(c, Ap, A)
    ExaModels.@add_par(c, Dp, D)

    # G(i), i=1..29: (p'(t_i) - p(t_i)^2 - 1)^2
    ExaModels.@add_obj(
        c,
        (
            # derivative p'(t_i) = sum_{j=1}^{12} Dp[12*(i-1)+j] * x[j]
            Dp[N*(i-1)+1]*x[1] + Dp[N*(i-1)+2]*x[2] + Dp[N*(i-1)+3]*x[3] +
            Dp[N*(i-1)+4]*x[4] + Dp[N*(i-1)+5]*x[5] + Dp[N*(i-1)+6]*x[6] +
            Dp[N*(i-1)+7]*x[7] + Dp[N*(i-1)+8]*x[8] + Dp[N*(i-1)+9]*x[9] +
            Dp[N*(i-1)+10]*x[10] + Dp[N*(i-1)+11]*x[11] + Dp[N*(i-1)+12]*x[12] -
            # minus p(t_i)^2
            (Ap[N*(i-1)+1]*x[1] + Ap[N*(i-1)+2]*x[2] + Ap[N*(i-1)+3]*x[3] +
             Ap[N*(i-1)+4]*x[4] + Ap[N*(i-1)+5]*x[5] + Ap[N*(i-1)+6]*x[6] +
             Ap[N*(i-1)+7]*x[7] + Ap[N*(i-1)+8]*x[8] + Ap[N*(i-1)+9]*x[9] +
             Ap[N*(i-1)+10]*x[10] + Ap[N*(i-1)+11]*x[11] + Ap[N*(i-1)+12]*x[12])^2 -
            T(1)
        )^2 for i in 1:29
    )

    # G(30): x[1]^2
    ExaModels.@add_obj(c, x[1]^2)

    # G(31): (x[2] - x[1]^2 - 1)^2
    ExaModels.@add_obj(c, (x[2] - x[1]^2 - T(1))^2)

    return ExaModels.ExaModel(c; kwargs...)
end
