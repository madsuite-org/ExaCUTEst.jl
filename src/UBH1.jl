function ubh1_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    # UBH1: bring vehicle from initial state to origin in fixed time
    # 6 state variables Y(1..6,0..N), 3 control variables U(1..3,0..N)
    # NV = 9*(N+1) variables; NC = 6*N equality constraints
    # Variable layout (1-indexed):
    #   Y(I,t) for I=1..6, t=0..N: index (I-1)*(N+1) + t + 1
    #   U(I,t) for I=1..3, t=0..N: index 6*(N+1) + (I-1)*(N+1) + t + 1
    # Objective: (K/2) * sum_{t=1..N} sum_{I=1..3} (U(I,t-1)^2 + U(I,t)^2)
    # Constraints S(I,t) for I=1..6, t=1..N:
    #   I=1..3: Y(I,t) - Y(I,t-1) - (K/2)*(Y(I+3,t-1) + Y(I+3,t)) = 0
    #   I=4..6: Y(I,t) - Y(I,t-1) - (K/2)*(U(I-3,t-1) + U(I-3,t)) = 0

    T0 = 0.0; TF = 1000.0
    K  = (TF - T0) / N
    Kh = K / 2.0

    NP1 = N + 1
    NV  = 9 * NP1

    # Variable index functions (1-based)
    y_idx(I, t) = (I-1)*NP1 + t + 1         # I=1..6, t=0..N
    u_idx(I, t) = 6*NP1 + (I-1)*NP1 + t + 1 # I=1..3, t=0..N

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    x0   = zeros(T, NV)

    # Control bounds: U(I,t) ∈ [-1, 1]
    for i in 1:3, t in 0:N
        lvar[u_idx(i, t)] = T(-1); uvar[u_idx(i, t)] = T(1)
    end

    # Fixed initial conditions: Y(1..6, 0)
    init_y = [1000.0, 1000.0, 1000.0, -10.0, 10.0, -10.0]
    for i in 1:6
        lvar[y_idx(i, 0)] = T(init_y[i])
        uvar[y_idx(i, 0)] = T(init_y[i])
        x0[y_idx(i, 0)]   = T(init_y[i])
    end

    # Fixed final conditions: Y(1..6, N) = 0
    for i in 1:6
        lvar[y_idx(i, N)] = T(0); uvar[y_idx(i, N)] = T(0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: (K/2) * sum_{t=1..N} sum_{i=1..3} (u(i,t-1)^2 + u(i,t)^2)
    # = K/2 * (2*sum_interior + boundary) = K * sum_interior + K/2 * boundary
    # But directly: sum_{t=1..N, i=1..3} Kh*(u[i,t-1]^2 + u[i,t]^2)
    obj_data = [(ui0 = u_idx(i, t-1), ui1 = u_idx(i, t)) for t in 1:N for i in 1:3]
    ExaModels.@add_obj(c, T(Kh)*(x[d.ui0]^2 + x[d.ui1]^2) for d in obj_data)

    # Constraints S(I,t) for I=1..3, t=1..N:
    # Y(I,t) - Y(I,t-1) - Kh*(Y(I+3,t-1) + Y(I+3,t)) = 0
    con1_data = [(yi = y_idx(i, t), yi0 = y_idx(i, t-1),
                  yi3 = y_idx(i+3, t), yi30 = y_idx(i+3, t-1))
                 for t in 1:N for i in 1:3]
    ExaModels.@add_con(c,
        x[d.yi] - x[d.yi0] - T(Kh)*(x[d.yi30] + x[d.yi3])
        for d in con1_data)

    # Constraints S(I+3,t) for I=1..3, t=1..N:
    # Y(I+3,t) - Y(I+3,t-1) - Kh*(U(I,t-1) + U(I,t)) = 0
    con2_data = [(yi3 = y_idx(i+3, t), yi30 = y_idx(i+3, t-1),
                  ui = u_idx(i, t), ui0 = u_idx(i, t-1))
                 for t in 1:N for i in 1:3]
    ExaModels.@add_con(c,
        x[d.yi3] - x[d.yi30] - T(Kh)*(x[d.ui0] + x[d.ui])
        for d in con2_data)

    return ExaModels.ExaModel(c; kwargs...)
end

function ubh5_model(N = 500; T = Float64, backend = nothing, kwargs...)
    # UBH5: like UBH1 but with cumulative cost as extra state Y(7,t)
    # 7 state variables Y(1..7,0..N), 3 control variables U(1..3,0..N)
    # NV = 10*(N+1) variables; NC = 7*N equality constraints
    # Y(7,t) = cumulative sum: Y(7,t) - Y(7,t-1) - Kh*(sum_i u_i(t-1)^2 + u_i(t)^2) = 0
    # Objective: Y(7,N)

    T0 = 0.0; TF = 1000.0
    K  = (TF - T0) / N
    Kh = K / 2.0

    NP1 = N + 1
    NV  = 10 * NP1

    y_idx(I, t) = (I-1)*NP1 + t + 1         # I=1..7, t=0..N
    u_idx(I, t) = 7*NP1 + (I-1)*NP1 + t + 1 # I=1..3, t=0..N

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    x0   = zeros(T, NV)

    # Control bounds: U(I,t) ∈ [-1, 1]
    for i in 1:3, t in 0:N
        lvar[u_idx(i, t)] = T(-1); uvar[u_idx(i, t)] = T(1)
    end

    # Fixed initial conditions: Y(1..6, 0)
    init_y = [1000.0, 1000.0, 1000.0, -10.0, 10.0, -10.0]
    for i in 1:6
        lvar[y_idx(i, 0)] = T(init_y[i])
        uvar[y_idx(i, 0)] = T(init_y[i])
        x0[y_idx(i, 0)]   = T(init_y[i])
    end
    # Y(7,0) = 0 (cumulative starts at 0)
    lvar[y_idx(7, 0)] = T(0); uvar[y_idx(7, 0)] = T(0)

    # Fixed final conditions: Y(1..6, N) = 0
    for i in 1:6
        lvar[y_idx(i, N)] = T(0); uvar[y_idx(i, N)] = T(0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: Y(7, N)
    ExaModels.@add_obj(c, x[y_idx(7, N)])

    # Constraints S(I,t) for I=1..3, t=1..N:
    # Y(I,t) - Y(I,t-1) - Kh*(Y(I+3,t-1) + Y(I+3,t)) = 0
    con1_data = [(yi = y_idx(i, t), yi0 = y_idx(i, t-1),
                  yi3 = y_idx(i+3, t), yi30 = y_idx(i+3, t-1))
                 for t in 1:N for i in 1:3]
    ExaModels.@add_con(c,
        x[d.yi] - x[d.yi0] - T(Kh)*(x[d.yi30] + x[d.yi3])
        for d in con1_data)

    # Constraints S(I+3,t) for I=1..3, t=1..N:
    # Y(I+3,t) - Y(I+3,t-1) - Kh*(U(I,t-1) + U(I,t)) = 0
    con2_data = [(yi3 = y_idx(i+3, t), yi30 = y_idx(i+3, t-1),
                  ui = u_idx(i, t), ui0 = u_idx(i, t-1))
                 for t in 1:N for i in 1:3]
    ExaModels.@add_con(c,
        x[d.yi3] - x[d.yi30] - T(Kh)*(x[d.ui0] + x[d.ui])
        for d in con2_data)

    # Cumulative constraint S(7,t) for t=1..N:
    # Y(7,t) - Y(7,t-1) - Kh * sum_{i=1..3} (u_i(t-1)^2 + u_i(t)^2) = 0
    # Split into 3 terms per timestep: each adds -Kh*(u_i(t-1)^2 + u_i(t)^2)
    # But @add_con creates one constraint per item in loop.
    # Can't aggregate, so need to handle differently.
    # One constraint per (t,i) won't work; but S(7,t) is a sum.
    # Since the state Y(7,t) accumulates, the constraint is:
    #   Y(7,t) - Y(7,t-1) = Kh * sum_i (u_i^2(t-1) + u_i^2(t))
    # This is one constraint per t with 3 control pairs. Expand inline.
    cum_data = [(y7 = y_idx(7, t), y70 = y_idx(7, t-1),
                 u10 = u_idx(1, t-1), u11 = u_idx(1, t),
                 u20 = u_idx(2, t-1), u21 = u_idx(2, t),
                 u30 = u_idx(3, t-1), u31 = u_idx(3, t))
                for t in 1:N]
    ExaModels.@add_con(c,
        x[d.y7] - x[d.y70]
        - T(Kh)*(x[d.u10]^2 + x[d.u11]^2
               + x[d.u20]^2 + x[d.u21]^2
               + x[d.u30]^2 + x[d.u31]^2)
        for d in cum_data)

    return ExaModels.ExaModel(c; kwargs...)
end
