# KSIP: Semi-infinite QP (Kortanek and No)
# min sum_{j=1}^20 [x_j^2/(2j) + x_j/j]
# s.t. sum_{j=1}^20 t^{j-1} * x_j >= sin(t) for t in {0, 1/1000, ..., 1000/1000}
# Variables: 20 free, N=20, M=1000, MP1=1001
# Element PSQ(V1; P): P*V1^2
# Objective: sum E(j) = sum P_j * x_j^2 where P_j = 1/(2j)
# Linear objective: sum JINV * x_j where JINV = 1/j
# CON(i): sum_{j=1}^N t_i^{j-1} * x_j >= sin(t_i) (G type, lcon=rhs, ucon=Inf)
# t_i = (i-1)/1000 for i=1..1001 (CON(1) has t=0)
# Note: CON(1): t=0, sum = x_1 >= sin(0)=0 → x_1 >= 0
function ksip_model(; T = Float64, backend = nothing, kwargs...)
    N = 20
    M = 1000
    MP1 = M + 1

    # Constraint data: t_i = (i-1)/M for i=1..MP1
    # t_i^{j-1} for j=1..N, i=1..MP1
    # Group CON(i): sum_{j=1}^N t_i^{j-1} * x_j >= sin(t_i)
    ts = T[(i - 1) / M for i in 1:MP1]
    rhs = T[sin(ts[i]) for i in 1:MP1]

    # Precompute t^{j-1} coefficients matrix: coeff[i,j] = t_i^{j-1}
    coeff = T[ts[i]^(j-1) for i in 1:MP1, j in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # All variables are free (FR 'DEFAULT')
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(2))

    # Objective: sum_{j=1}^N [PSQ(x_j; 1/(2j)) + (1/j)*x_j]
    # PSQ element: P * V1^2, P = 1/(2j) → contribution = x_j^2/(2j)
    # Linear: JINV * x_j where JINV = 1/j
    ExaModels.@add_obj(c, (x[j]^2 / T(2 * j) + x[j] / T(j)) for j in 1:N)

    # Constraints CON(i) for i=1..MP1: sum_{j=1}^N t_i^{j-1} * x_j >= sin(t_i)
    # i.e., sum_{j=1}^N coeff[i,j] * x_j - sin(t_i) >= 0
    ExaModels.@add_par(c, RHS, rhs)
    ExaModels.@add_par(c, CF, T[coeff[i, j] for i in 1:MP1 for j in 1:N])

    # Create one constraint per t value:
    # @add_con creates constraints via loop; expression involves sum over j=1..N
    # Store coefficients as a 2D parameter array indexed by (i,j)
    # Actually the coefficients coeff[i,j] are plain Julia values (not parameters),
    # so we can use them directly in the expression.
    ExaModels.@add_con(c,
        sum(coeff[i, j] * x[j] for j in 1:N) - RHS[i]
        for i in 1:MP1;
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
