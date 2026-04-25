# SPARSQUR: sparse quartic problem (N = 10000 variables, unconstrained)
# minimize sum_{i=1}^{N} 0.5 * i * (sum_j 0.5*X(J)^2)^2
#   where J ranges over {i, mod(2i-1,N)+1, mod(3i-1,N)+1, mod(5i-1,N)+1,
#                         mod(7i-1,N)+1, mod(11i-1,N)+1}
# All variables free (FR DEFAULT), start = 0.5
# SIF: GROUP TYPE SQR: F = 0.5 * P * ALPHA^2, where P = i (real index), ALPHA = group value
#      Element SQUR: F = 0.5 * X^2 (coefficient 1.0 in GROUP USES)
#      Group value: alpha_i = sum of coeff * element_value = sum_j 0.5 * X(J)^2
#      Contribution: 0.5 * i * alpha_i^2 = 0.5 * i * (0.5 * sum_j X(J)^2)^2
#                  = (i/8) * (sum_j X(J)^2)^2
# Source: Nick Gould, November 1995
function sparsqur_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    # Precompute the 6 index tuples for each group i
    idx = [(
        i,
        mod(2 * i - 1, N) + 1,
        mod(3 * i - 1, N) + 1,
        mod(5 * i - 1, N) + 1,
        mod(7 * i - 1, N) + 1,
        mod(11 * i - 1, N) + 1,
    ) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0.5))
    ExaModels.@add_par(c, p, [T(i) for i in 1:N])
    ExaModels.@add_par(c, q, idx)
    ExaModels.@add_obj(
        c,
        T(0.5) * p[i] * (T(0.5) * (x[q[i][1]]^2 + x[q[i][2]]^2 + x[q[i][3]]^2 +
                                    x[q[i][4]]^2 + x[q[i][5]]^2 + x[q[i][6]]^2))^2
        for i in 1:N
    )
    return ExaModels.ExaModel(c; kwargs...)
end
