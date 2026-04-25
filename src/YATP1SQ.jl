# YATP1SQ: Yet Another Test Problem with double pseudo-stochastic constraints
# N^2 + 2N variables: X(I,J) for I,J=1..N (N^2 vars), Y(I) and Z(I) for I=1..N (2N vars)
# All variables free (FR DEFAULT).
# Variable ordering (row-major X, interleaved Y/Z):
#   x[k] = X(I,J) for k = (I-1)*N + J   (I,J = 1..N)
#   x[N^2 + 2*(I-1) + 1] = Y(I)  for I = 1..N
#   x[N^2 + 2*(I-1) + 2] = Z(I)  for I = 1..N
#
# Constraints (N^2 + 2N equalities, all lcon=ucon=0):
#   E(I,J) = X^3 - A*X^2 - (Y(I)+Z(I)) * X * cos(X) + (Y(I)+Z(I)) * sin(X) = 0
#            where X = X(I,J), for I,J = 1..N  (constraints 1..N^2 in row-major order)
#   ER(I) = sum_{J=1}^N sin(X(I,J)) / X(I,J) - 1 = 0
#   EC(I) = sum_{J=1}^N sin(X(J,I)) / X(J,I) - 1 = 0
#   ER(I) and EC(I) are interleaved in CUTEst: ER(1), EC(1), ER(2), EC(2), ..., ER(N), EC(N)
#
# A = 10.0 (SIF default)
# Default N = 10 (testable with CUTEst; original SIF default is N=350)
# Source: Ph. Toint, June 2003. SIF: NOR2-AN-V-V
function yatp1sq_model(N = 10; T = Float64, backend = nothing, kwargs...)
    A = T(10)
    nvar = N^2 + 2 * N

    start = vcat(fill(T(6), N^2), fill(T(0), 2 * N))
    ij_pairs = [(i, j) for i in 1:N for j in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    ExaModels.@add_var(c, x, nvar;
        lvar  = fill(T(-Inf), nvar),
        uvar  = fill(T(Inf), nvar),
        start = start)

    # E(I,J) constraints in row-major order: constraint (I-1)*N+J
    # x_{ij}^3 - A*x_{ij}^2 - (y_i+z_i)*x_{ij}*cos(x_{ij}) + (y_i+z_i)*sin(x_{ij}) = 0
    ExaModels.@add_con(c,
        begin
            xij = x[(i - 1) * N + j]
            yi  = x[N * N + 2 * (i - 1) + 1]
            zi  = x[N * N + 2 * (i - 1) + 2]
            xij^3 - A * xij^2 - (yi + zi) * xij * cos(xij) + (yi + zi) * sin(xij)
        end
        for (i, j) in ij_pairs)

    # ER(I) and EC(I) interleaved: ER(1), EC(1), ER(2), EC(2), ..., ER(N), EC(N)
    for i in 1:N
        # ER(I): sum_{j=1..N} sin(X(I,j)) / X(I,j) - 1 = 0
        ExaModels.@add_con(c,
            sum(sin(x[(i - 1) * N + j]) / x[(i - 1) * N + j] for j in 1:N) - T(1))
        # EC(I): sum_{j=1..N} sin(X(j,I)) / X(j,I) - 1 = 0
        ExaModels.@add_con(c,
            sum(sin(x[(j - 1) * N + i]) / x[(j - 1) * N + i] for j in 1:N) - T(1))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
