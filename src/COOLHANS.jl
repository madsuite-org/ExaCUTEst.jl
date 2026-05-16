function coolhans_model(; T = Float64, backend = nothing, kwargs...)
    A = T[0.0        0.0      0.0;
          1.3725e-7  937.62  -42.207;
          0.0        0.0      0.0]
    B = T[0.0060893   -44.292    2.0011;
          1.3880e-7  -1886.0    42.362;
         -1.3877e-7   42.362   -2.0705]
    C = T[0.0      44.792    0.0;
          0.0     948.21     0.0;
          0.0     -42.684    0.0]

    vidx(i, j) = (i - 1) * 3 + j

    data = [(
        b1  = B[K, 1], b2  = B[K, 2], b3  = B[K, 3],
        a1  = A[K, 1], a2  = A[K, 2], a3  = A[K, 3],
        m1L = vidx(1, L), m2L = vidx(2, L), m3L = vidx(3, L),
        c0  = C[K, L],
    ) for K in 1:3 for L in 1:3]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 9; lvar = fill(T(-Inf), 9), uvar = fill(T(Inf), 9), start = zeros(T, 9))
    ExaModels.@add_con(c,
        d.b1 * x[d.m1L] + d.b2 * x[d.m2L] + d.b3 * x[d.m3L]
        + d.a1 * (x[1] * x[d.m1L] + x[2] * x[d.m2L] + x[3] * x[d.m3L])
        + d.a2 * (x[4] * x[d.m1L] + x[5] * x[d.m2L] + x[6] * x[d.m3L])
        + d.a3 * (x[7] * x[d.m1L] + x[8] * x[d.m2L] + x[9] * x[d.m3L])
        + d.c0 for d in data)
    return ExaModels.ExaModel(c; kwargs...)
end
