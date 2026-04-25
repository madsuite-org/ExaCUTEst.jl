# HILBERTA: Hilbert quadratic (N=2 default, scalable)
# minimize sum_{I=1}^{N} sum_{J=1}^{I} G(I,J)
#
# SIF structure (N=2 default):
#   Groups G(I,J) for 1 <= J <= I <= N, all N-type (objective), no group type (identity)
#   Element types: 2PR(X,Y) = X*Y, SQ(X) = X^2
#
#   For J < I: contribution = HIJ * X(I)*X(J)   where HIJ = 1/(I+J-1)
#   For J = I: contribution = COEFF * X(I)^2    where COEFF = HII/2 + D (D=0)
#              COEFF = 0.5/(2I-1) = 1/(2*(2I-1))
#
# All variables free (FR DEFAULT), start = -3.0
function hilberta_model(N = 2; T = Float64, backend = nothing, kwargs...)
    # Off-diagonal pairs (J < I) with coefficients HIJ = 1/(I+J-1)
    od_data = [(i, j, T(1 / (i + j - 1))) for i in 1:N for j in 1:i-1]
    # Diagonal data: (i, COEFF_i) where COEFF_i = 1/(2*(2I-1))
    d_data  = [(i, T(1 / (2 * (2 * i - 1)))) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(-3.0))

    # Diagonal terms: COEFF_i * X(i)^2
    ExaModels.@add_obj(c, coef * x[i]^2 for (i, coef) in d_data)

    # Off-diagonal terms: HIJ * X(I)*X(J)
    if !isempty(od_data)
        ExaModels.@add_obj(c, coef * x[i] * x[j] for (i, j, coef) in od_data)
    end

    return ExaModels.ExaModel(c; kwargs...)
end
