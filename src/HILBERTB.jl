# HILBERTB: Perturbed Hilbert quadratic (N=10, all free, start=-3.0, D=5.0)
# G(I,J) groups for I=1..N, J=1..I — identity group type (contribution = v directly)
# Off-diagonal (J < I): element 2PR(X(I),X(J)) = X(I)*X(J) with coefficient 1/(I+J-1)
#   contribution = X(I)*X(J) / (I+J-1)
# Diagonal (J = I): element SQ(X(I)) = X(I)^2 with coefficient HII/2 + D = 1/(2*(2I-1)) + 5
#   contribution = (1/(2*(2I-1)) + D) * X(I)^2
function hilbertb_model(N = 10; T = Float64, backend = nothing, kwargs...)
    D = T(5.0)
    data_offdiag = [(i = i, j = j, coeff = T(1) / T(i + j - 1)) for i in 1:N for j in 1:i-1]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(-3))
    ExaModels.@add_par(c, DIAG, T[T(1) / (T(2) * T(2*i - 1)) + D for i in 1:N])
    ExaModels.@add_obj(c, d.coeff * x[d.i] * x[d.j] for d in data_offdiag)
    ExaModels.@add_obj(c, DIAG[i] * x[i]^2 for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
