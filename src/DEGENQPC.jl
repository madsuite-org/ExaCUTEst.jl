# DEGENQPC: Degenerate convex QP, N=50
# N vars x(1..N), bounds [0,1], start=2
# Objective: sum_i x_i + sum_i (i/N)*0.5*x_i^2
# E-constraints: x(I) - x(I+1) = 0 for odd I (i=1,3,5,...,N-1)
# G-constraints with RANGE=2: x(i)+x(j)+x(k) in [0,2] for 1<=i<j<k<=N
function degenqpc_model(N = 50; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0), uvar = T(1), start = T(2))

    # Objective: linear + quadratic terms
    obj_data = [(i = i, p = T(i) / T(N) * T(0.5)) for i in 1:N]
    ExaModels.@add_obj(c, x[d.i] + d.p * x[d.i]^2 for d in obj_data)

    # E-constraints: x(I) - x(I+1) = 0 for odd I = 1, 3, 5, ..., N-1
    eq_data = [(i = i, ip1 = i + 1) for i in 1:2:N-1]
    ExaModels.@add_con(c, x[d.i] - x[d.ip1] for d in eq_data)

    # G-constraints with RANGE 2.0: x(i)+x(j)+x(k) in [0,2]
    ineq_data = [(i = i, j = j, k = k) for i in 1:N for j in i+1:N for k in j+1:N]
    ExaModels.@add_con(c, x[d.i] + x[d.j] + x[d.k]
        for d in ineq_data; lcon = T(0), ucon = T(2))

    return ExaModels.ExaModel(c; kwargs...)
end
