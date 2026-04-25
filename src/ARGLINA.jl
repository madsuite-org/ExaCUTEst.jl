# ARGLINA: Variable dimension full rank linear problem (N=200 variables, M=400 groups)
# minimize sum_{i=1}^{M} v_i^2   (L2 group type, no scale → contribution = v^2)
# For i=1..N:   v_i = sum_j c_{ij}*x_j - 1  where c_{ij} = -2/M (j≠i), c_{ii} = 1-2/M
# For i=N+1..M: v_i = sum_j (-2/M)*x_j - 1  (M-N identical groups, multiplied by M-N)
# BOUNDS: FR DEFAULT → all variables free; START: DEFAULT = 1.0
# Source: More', Garbow, Hillstrom (1981), problem 32
function arglina_model(N = 200, M = 400; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf),
                                 uvar = T(Inf),
                                 start = T(1.0))

    a = T(-2) / T(M)   # -2/M = -0.01

    # Groups i=1..N: v_i = sum_j c_{ij}*x[j] - 1
    # c_{ij} = a for j≠i,  c_{ii} = 1+a = 1-2/M
    # Use per-group parameter arrays (length N) — same expression type, different data
    for i in 1:N
        coeff = T[j == i ? T(1) + a : a for j in 1:N]
        ExaModels.@add_par(c, Ci, coeff)
        ExaModels.@add_obj(c, (sum(Ci[j] * x[j] for j in 1:N) - T(1))^2)
    end

    # Groups i=N+1..M: identical, v = a*sum_j x[j] - 1, contribution = v^2 each
    # All M-N groups have the same expression, multiply by (M-N)
    ExaModels.@add_obj(c, T(M - N) * (a * sum(x[j] for j in 1:N) - T(1))^2)

    return ExaModels.ExaModel(c; kwargs...)
end
