# CHANDHEQ: Chandrasekhar H-equation (More' 1988, problem 4)
# N=100 positive variables H(1)..H(N), N equality constraints
# G(I) = H(I) - 1 - 0.5*C * X(I) * sum_J (W(J)/(X(I)+X(J))) * H(I)*H(J) = 0
# where X(I)=I/N, W(J)=1/N, C=1 (parameter in [0,1])
# lvar=0 (positive), start=1
function chandheq_model(N = 100; T = Float64, backend = nothing, kwargs...)
    C = T(1)
    HALFC = C / T(2)
    inv_N = T(1) / T(N)
    # Precompute coefficients: coeff[i,j] = -HALFC * X(I)*W(J)/(X(I)+X(J))
    # X(I) = I/N, W(J) = 1/N → coeff = -HALFC * I/(N*(I+J))
    data_ij = [(i = i, j = j,
        coeff = -HALFC * T(i) * inv_N * inv_N / (T(i) * inv_N + T(j) * inv_N))
        for i in 1:N, j in 1:N][:]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0), uvar = T(Inf), start = T(1))
    ExaModels.@add_con(c, g, x[i] - T(1) for i in 1:N)
    ExaModels.@add_con!(c, g[d.i] += d.coeff * x[d.i] * x[d.j] for d in data_ij)
    return ExaModels.ExaModel(c; kwargs...)
end
