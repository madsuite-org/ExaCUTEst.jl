# CHANDHEU: Chandrasekhar H-equation (free variable variant)
# N=500 free variables, same structure as CHANDHEQ but lvar=-Inf
function chandheu_model(N = 500; T = Float64, backend = nothing, kwargs...)
    C = T(1)
    HALFC = C / T(2)
    inv_N = T(1) / T(N)
    data_ij = [(i = i, j = j,
        coeff = -HALFC * T(i) * inv_N * inv_N / (T(i) * inv_N + T(j) * inv_N))
        for i in 1:N, j in 1:N][:]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1))
    ExaModels.@add_con(c, g, x[i] - T(1) for i in 1:N)
    ExaModels.@add_con!(c, g[d.i] += d.coeff * x[d.i] * x[d.j] for d in data_ij)
    return ExaModels.ExaModel(c; kwargs...)
end
