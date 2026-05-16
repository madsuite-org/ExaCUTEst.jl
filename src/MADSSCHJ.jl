# MADSSCHJ: nonlinear minmax, N=200 default, M=2N-2 constraints
# Variables: X(1)..X(N), Z; Z is last variable
# Objective: min Z (linear)
# Constraints (G type, >= 0 → lcon=0):
# C(k) >= 0 for k=1..M where each is Z - sum_{i in S(k)} X(i) - X(k_special)^2 >= ...
# After analysis:
# C(1): Z - sum_{i=2..N} X(i) - X(1)^2 >= -1 → C(1): Z - sum_{i=2..N} X(i) - X(1)^2 - (-1) >= 0
# C(2): Z - X(1) - sum_{i=3..N} X(i) - X(2)^2 >= -1
# C(3): Z - X(1) - sum_{i=3..N} X(i) - 2*X(2)^2 >= -1
# For K=4..M-1 by pairs (DI 2):
#   J = (K+2)/2; C(K): Z - sum_{i neq J} X(i) - X(J)^2 >= -1
#                C(K+1): Z - sum_{i neq J} X(i) - 2*X(J)^2 >= -1
# C(M): Z - sum_{i=1..N-1} X(i) - X(N)^2 >= -1
# All free, start Z=0, X(i)=10
function madsschj_model(N = 200; T = Float64, backend = nothing, kwargs...)
    M = 2 * N - 2
    # Variables: x[1..N] = X(1..N), x[N+1] = Z
    n_vars = N + 1
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, n_vars; lvar = T(-Inf), uvar = T(Inf),
                       start = T[fill(10.0, N); 0.0])

    # Objective: min Z = x[N+1]
    ExaModels.@add_obj(c, x[N+1])

    # Sum of all X(i), i=1..N
    # We'll build constraints as: Z + rhs - sum_specific >= 0
    # C(1): Z - sum_{i=2..N} X(i) - X(1)^2 + 1 >= 0
    # = x[N+1] - sum_{i=2}^{N} x[i] - x[1]^2 + 1 >= 0
    ExaModels.@add_con(c, x[N+1] - sum(x[i] for i in 2:N) - x[1]^2 + T(1);
                       lcon = T(0), ucon = T(Inf))

    # C(2): Z - X(1) - sum_{i=3..N} X(i) - X(2)^2 + 1 >= 0
    ExaModels.@add_con(c, x[N+1] - x[1] - sum(x[i] for i in 3:N) - x[2]^2 + T(1);
                       lcon = T(0), ucon = T(Inf))

    # C(3): same sum as C(2) but -2*X(2)^2
    ExaModels.@add_con(c, x[N+1] - x[1] - sum(x[i] for i in 3:N) - T(2)*x[2]^2 + T(1);
                       lcon = T(0), ucon = T(Inf))

    # DO K=4..M-1 by 2: J = (K+2)/2
    # C(K): Z - sum_{i neq J, i=1..N} X(i) - X(J)^2 + 1 >= 0
    # C(K+1): same but -2*X(J)^2
    mid_data = [(J = (K + 2) ÷ 2,) for K in 4:2:M-1]
    ExaModels.@add_con(c,
        x[N+1] - sum(x[i] for i in 1:N) + x[d.J] - x[d.J]^2 + T(1)
        for d in mid_data; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c,
        x[N+1] - sum(x[i] for i in 1:N) + x[d.J] - T(2)*x[d.J]^2 + T(1)
        for d in mid_data; lcon = T(0), ucon = T(Inf))

    # C(M): Z - sum_{i=1..N-1} X(i) - X(N)^2 + 1 >= 0
    ExaModels.@add_con(c, x[N+1] - sum(x[i] for i in 1:N-1) - x[N]^2 + T(1);
                       lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
