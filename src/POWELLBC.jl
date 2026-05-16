# POWELLBC: Minimize sum_{k>j} 1/||p_j - p_k||_2
# Points p_j = (x_{2j-1}, x_{2j}) for j=1..P
# INVNRM element: F = 1/sqrt((XJ-XK)^2 + (YJ-YK)^2)
# Bounds: 0 <= x_i <= 1
# Start: x_i = (i/N)^2
function powellbc_model(P = 500; T = Float64, backend = nothing, kwargs...)
    N = 2 * P
    RN = T(N)

    lvar = zeros(T, N)
    uvar = ones(T, N)

    start = Vector{T}(undef, N)
    for i in 1:N
        start[i] = (T(i) / RN)^2
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # Objective: sum_{k=2}^{P} sum_{j=1}^{k-1} 1/sqrt((x_{2j-1}-x_{2k-1})^2 + (x_{2j}-x_{2k})^2)
    for k in 2:P
        xk_idx = 2 * k - 1
        yk_idx = 2 * k
        for j in 1:k-1
            xj_idx = 2 * j - 1
            yj_idx = 2 * j
            ExaModels.@add_obj(c, T(1.0) / sqrt((x[xj_idx] - x[xk_idx])^2 + (x[yj_idx] - x[yk_idx])^2))
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end
