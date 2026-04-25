# HILBERTA: Hilbert quadratic (N=2 default, scalable)
# minimize sum_{I=1}^{N} sum_{J=1}^{I} G(I,J)
#
# SIF structure (N=2 default):
#   Groups G(I,J) for 1 <= J <= I <= N, all N-type (objective), no group type (identity)
#   Element types: 2PR(X,Y) = X*Y, SQ(X) = X^2
#
#   For J < I: contribution = HIJ * X(I)*X(J)   where HIJ = 1/(I+J-1)
#   For J = I: contribution = COEFF * X(I)^2    where COEFF = 1/(2*(2I-1))
#
# All variables free (FR DEFAULT), start = -3.0
function hilberta_model(N = 2; T = Float64, backend = nothing, kwargs...)
    # Off-diagonal pairs (J < I) with coefficients
    od_pairs = [(i, j, T(1 / (i + j - 1))) for i in 1:N for j in 1:i-1]
    # Diagonal coefficients: COEFF[i] = 1/(2*(2I-1))
    d_coef = T[1 / (2 * (2 * i - 1)) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(-3.0))

    # Diagonal terms: COEFF[i] * X(i)^2
    ExaModels.@add_par(c, dcoef, d_coef)
    ExaModels.@add_obj(c, dcoef[i] * x[i]^2 for i in 1:N)

    # Off-diagonal terms: HIJ * X(I)*X(J)
    if !isempty(od_pairs)
        # Extract coefficients into a parameter array; use pair (i,j) as integer indices
        od_c = T[coef for (_, _, coef) in od_pairs]
        ExaModels.@add_par(c, odcoef, od_c)
        ExaModels.@add_obj(
            c,
            odcoef[k] * x[od_pairs[k][1]] * x[od_pairs[k][2]]
            for k in 1:length(od_pairs)
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end
