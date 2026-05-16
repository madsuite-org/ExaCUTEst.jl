# S268: Quadratic programming problem with 5 variables and 5 linear constraints
# Source: Schittkowski "More Test Examples for NLP Codes", problem 268
# Objective: sum_{i,j} D(i,j)*x_i*x_j - 14463 + sum_i B(i)*x_i
# Constraints (G type, >= rhs): each row of A, x >= b
# Bounds: XR 'DEFAULT' -> all variables free
function s268_model(; T = Float64, backend = nothing, kwargs...)
    # D matrix entries as flat list
    D_vals = T[10197.0, -12454.0, -1013.0, 1948.0, 329.0,
               -12454.0, 20909.0, -1733.0, -4914.0, -186.0,
               -1013.0, -1733.0, 1755.0, 1089.0, -174.0,
               1948.0, -4914.0, 1089.0, 1515.0, -22.0,
               329.0, -186.0, -174.0, -22.0, 27.0]
    B_vals = T[-9170.0, 17099.0, -2271.0, -4336.0, -43.0]
    # Constraint rows: A[i,j], rhs b[i] (G type: Ax - b >= 0)
    A_vals = T[-1.0, -1.0, -1.0, -1.0, -1.0,
               10.0, 10.0, -3.0, 5.0, 4.0,
               -8.0, 1.0, -2.0, -5.0, 3.0,
               8.0, -1.0, 2.0, 5.0, -3.0,
               -4.0, -2.0, 3.0, -5.0, 1.0]
    b_vals = T[-5.0, 20.0, -40.0, 11.0, -30.0]

    # index pairs for the 25 D terms
    ij_data = [(i = i, j = j, k = (i - 1) * 5 + j) for i in 1:5 for j in 1:5]
    con_data = [(i = i,) for i in 1:5]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T(-Inf), uvar = T(Inf), start = T(1))

    # Objective: NONL group: sum_ij D(i,j)*x_i*x_j - 14463
    ExaModels.@add_par(c, Dij, D_vals)
    ExaModels.@add_obj(c, Dij[d.k] * x[d.i] * x[d.j] for d in ij_data)
    ExaModels.@add_obj(c, -T(14463))
    # Objective: LINEAR group: sum_i B(i)*x_i
    ExaModels.@add_par(c, Bi, B_vals)
    ExaModels.@add_obj(c, Bi[i] * x[i] for i in 1:5)

    # Constraints: G type, Ax >= b -> Ax - b >= 0
    ExaModels.@add_par(c, Aij, A_vals)
    ExaModels.@add_par(c, bi, b_vals)
    con2_data = [(i = i, j1 = (i - 1) * 5 + 1) for i in 1:5]
    ExaModels.@add_con(c,
        Aij[d.j1] * x[1] + Aij[d.j1 + 1] * x[2] + Aij[d.j1 + 2] * x[3] +
        Aij[d.j1 + 3] * x[4] + Aij[d.j1 + 4] * x[5] - bi[d.i]
        for d in con2_data; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
