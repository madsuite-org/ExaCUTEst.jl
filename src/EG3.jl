# EG3: N=10000 variables X(1..N) + Y; N-1 L-type, N G-type, 1 E-type constraint
# CUTEst variable ordering: Y(1), X(1)(2), X(2)(3), X(N)(4), X(4..N-1)(5..N), X(3)(N+1)
# Constraint ordering: Q(E) at 1, Q(2..N-1)(L) at 2..N-1, S(1..N)(G+range) at N..2N-1, Q(1)(L) at 2N
function eg3_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    NV = N + 1
    Y_IDX = 1

    # CUTEst variable ordering: Y, X(1), X(2), X(N), X(4..N-1), X(3)
    X_pos(I) = if I == 1; 2
               elseif I == 2; 3
               elseif I == 3; N + 1
               elseif I == N; 4
               else; I + 1  # I = 4..N-1
               end

    # Build bounds in CUTEst order
    lvar = fill(T(-1), NV)
    lvar[Y_IDX] = T(-Inf)
    uvar = fill(T(-Inf), NV)
    uvar[Y_IDX] = T(Inf)
    uvar[2] = T(1)     # X(1)
    uvar[3] = T(2)     # X(2)
    uvar[4] = T(N)     # X(N)
    for k in 5:N
        uvar[k] = T(k - 1)  # X(k-1) for k=5..N
    end
    uvar[N+1] = T(3)   # X(3)

    x0 = fill(T(0.5), NV)
    x0[Y_IDX] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: HALFL2(Y + (X(1)-X(N))*X(2)) = 0.5*(Y + (X(1)-X(N))*X(2))^2
    ExaModels.@add_obj(c, T(0.5) * (x[Y_IDX] + (x[X_pos(1)] - x[X_pos(N)]) * x[X_pos(2)])^2)

    # Constraint 1: Q (E-type): (X(1)+X(N))^2 - 1 = 0
    ExaModels.@add_con(c, (x[X_pos(1)] + x[X_pos(N)])^2 - T(1))

    # Constraints 2..N-1: Q(I) for I=2..N-1 (L-type ≤ 0)
    # Y + X(1)*X(I+1) + (1+2/I)*X(I)*X(N) ≤ 0
    q_data = [(xi = Int32(X_pos(i)), xip1 = Int32(X_pos(i+1)), coeff = T(1 + 2.0/i)) for i in 2:N-1]
    ExaModels.@add_con(c, x[Y_IDX] + x[X_pos(1)] * x[d.xip1] + d.coeff * x[d.xi] * x[X_pos(N)]
        for d in q_data; lcon = T(-Inf), ucon = T(0))

    # Constraints N..2N-1: S(I) for I=1..N (G-type with RANGES=0.5)
    s_data = [(xi = Int32(X_pos(i)),) for i in 1:N]
    ExaModels.@add_con(c, sin(x[d.xi])^2 for d in s_data; lcon = T(0), ucon = T(0.5))

    # Constraint 2N: Q(1) (L-type ≤ 0)
    # Y + X(1)*X(2) + 3*X(1)*X(N) ≤ 0
    ExaModels.@add_con(c, x[Y_IDX] + x[X_pos(1)] * x[X_pos(2)] + T(3) * x[X_pos(1)] * x[X_pos(N)];
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
