# SPARSQUR: sparse quartic problem (N = 10000 variables, unconstrained)
# minimize sum_{i=1}^{N} 0.5 * i * alpha_i^2
#   where alpha_i = sum_j 0.5 * X(J)^2  (element SQUR: F = 0.5*X^2, coeff 1.0)
#   J ranges over {i, j1(i), j2(i), j3(i), j4(i), j5(i)}:
#     j1 = mod(2i-1, N)+1, j2 = mod(3i-1, N)+1, j3 = mod(5i-1, N)+1,
#     j4 = mod(7i-1, N)+1, j5 = mod(11i-1, N)+1
# Group value: alpha_i = 0.5*(x[i]^2 + x[j1]^2 + x[j2]^2 + x[j3]^2 + x[j4]^2 + x[j5]^2)
# Contribution: 0.5 * i * alpha_i^2 = (i/8)*(sum_j x[J]^2)^2
# All variables free (FR DEFAULT), start = 0.5
# Default N = 10000
# Source: Nick Gould, November 1995
function sparsqur_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    # Precompute index 7-tuples: (i, i, j1, j2, j3, j4, j5)
    # First element is the weight, remaining 6 are variable indices
    data = [
        (
            Float64(i),
            i,
            mod(2 * i - 1, N) + 1,
            mod(3 * i - 1, N) + 1,
            mod(5 * i - 1, N) + 1,
            mod(7 * i - 1, N) + 1,
            mod(11 * i - 1, N) + 1,
        ) for i in 1:N
    ]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0.5))
    ExaModels.@add_obj(
        c,
        T(0.5) * wi * (T(0.5) * (x[i0]^2 + x[j1]^2 + x[j2]^2 +
                                   x[j3]^2 + x[j4]^2 + x[j5]^2))^2
        for (wi, i0, j1, j2, j3, j4, j5) in data
    )
    return ExaModels.ExaModel(c; kwargs...)
end
