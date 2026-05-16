# KSS: Nonlinear system with exponential multiplicity at zero
# N=1000 variables (default), equality constraints
# E(i) = sum_{j=1}^N x_j - 3*x_i + x_i^2 - (N-1) = 0
# i.e., E(i) = sum_{j!=i} x_j - 2*x_i + x_i^2 - (N-1) = 0
# Or equivalently: (sum_{j=1}^N x_j) - 4*x_i + x_i^2 = N-1
# No objective (purely equality system)
# Start: x_i = 1000 for all i
# Variables: free
function kss_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    RN_minus1 = T(N - 1)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1000))

    # E(i): sum_{j=1}^{i-1} x_j - 3*x_i + sum_{j=i+1}^N x_j + x_i^2 - (N-1) = 0
    # = (sum_{j=1}^N x_j) - 4*x_i + x_i^2 - (N-1) = 0
    # Linear part from GROUPS: for each i, sum_{j!=i} x_j with coeff 1, x_i with coeff -3
    # Plus element SQ(x_i) with coeff 1 → x_i^2
    # Constant rhs = N-1
    # Net: sum_{j=1}^N x_j - 4*x_i + x_i^2 = N-1
    # In residual form: sum_{j=1}^N x_j - 4*x_i + x_i^2 - (N-1) = 0
    # Note: SIF E(i) group has linear part x_i with coeff -3, and sum_{j!=i} x_j with coeff 1
    # Plus element S(i)=SQ(x_i)=x_i^2
    # E(i) = -3*x_i + sum_{j!=i} x_j + x_i^2 = N-1
    # = sum_j x_j - 4*x_i + x_i^2 = N-1

    ExaModels.@add_con(c,
        sum(x[j] for j in 1:N) - T(4) * x[i] + x[i]^2 - RN_minus1
        for i in 1:N)

    return ExaModels.ExaModel(c; kwargs...)
end
