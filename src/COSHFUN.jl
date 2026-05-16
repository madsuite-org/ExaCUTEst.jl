function coshfun_model(M = 2000; T = Float64, backend = nothing, kwargs...)
    # N = 3*M independent variables + 1 variable F
    N  = 3 * M
    NV = N + 1   # total variables; F is at index NV

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = fill(T(-Inf), NV), uvar = fill(T(Inf), NV), start = zeros(T, NV))

    # Objective: minimize F = x[NV]
    ExaModels.@add_obj(c, x[NV])

    # Element functions: for triplet j (j=1..M), using X(3j-2), X(3j-1), X(3j):
    #   SQR(j) = X(3j)^2
    #   COSH(j) = cosh(X(3j-1))
    #   PROD(j) = 2*X(3j-2)^2*X(3j)
    # All added with coefficient 1.0 to C(j).
    #
    # Linear terms:
    #   C(1):      -F - 2*X(3) - X(6)
    #   C(j) 2..M-1: -F + X(3j-5) - 2*X(3j) - X(3j+3)
    #   C(M):      -F + X(3M-5) - 2*X(3M)

    # C(1) — scalar (j=1)
    ExaModels.@add_con(c,
        -x[NV] - T(2) * x[3] - x[6]
        + x[3]^2 + cosh(x[2]) + T(2) * x[1]^2 * x[3];
        lcon = T(-Inf), ucon = T(0))

    # C(j) for j=2..M-1
    if M > 2
        b_data = [(j1 = 3j - 5, j2 = 3j, j3 = 3j + 3,
                   sq = 3j, cs = 3j - 1, pr1 = 3j - 2, pr2 = 3j) for j in 2:M-1]
        ExaModels.@add_con(c,
            -x[NV] + x[d.j1] - T(2) * x[d.j2] - x[d.j3]
            + x[d.sq]^2 + cosh(x[d.cs]) + T(2) * x[d.pr1]^2 * x[d.pr2]
            for d in b_data; lcon = T(-Inf), ucon = T(0))
    end

    # C(M) — scalar
    ExaModels.@add_con(c,
        -x[NV] + x[3M - 5] - T(2) * x[3M]
        + x[3M]^2 + cosh(x[3M - 1]) + T(2) * x[3M - 2]^2 * x[3M];
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
