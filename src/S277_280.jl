# S277-280: LP based on the Hilbert matrix (N=4 default = problem 277)
# min  sum_i (sum_j 1/(i+j-1)) * x_i
# s.t. sum_j 1/(i+j-1) * x_j >= sum_j 1/(i+j-1) (i.e. same Hilbert matrix used for RHS)
# Bounds: x >= 0 (default, no bounds section)
function s277_280_model(N = 4; T = Float64, backend = nothing, kwargs...)
    # Hilbert matrix H[i,j] = 1/(i+j-1)
    H = T[1 / (i + j - 1) for i in 1:N, j in 1:N]
    # Objective coefficients: c[i] = sum_j H[i,j]
    obj_coeff = T[sum(H[i, :]) for i in 1:N]
    # Constraint RHS: b[i] = sum_j H[i,j] = obj_coeff[i]
    b_vals = obj_coeff

    con_data = [(i = i,) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; start = T(0))
    # Objective: sum_i obj_coeff[i] * x[i]
    ExaModels.@add_par(c, ci, obj_coeff)
    ExaModels.@add_obj(c, ci[i] * x[i] for i in 1:N)
    # Constraints: sum_j H[i,j]*x[j] >= b[i]  -> sum_j H[i,j]*x[j] - b[i] >= 0
    ExaModels.@add_par(c, Hij, T[H[d.i, j] for d in con_data for j in 1:N])
    ExaModels.@add_par(c, bi, b_vals)
    con2_data = [(i = i, j1 = (i - 1) * N + 1) for i in 1:N]
    ExaModels.@add_con(c,
        sum(Hij[d.j1 + j - 1] * x[j] for j in 1:N) - bi[d.i]
        for d in con2_data; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
