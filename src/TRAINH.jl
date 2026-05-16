function trainh_model(N = 1001; T = Float64, backend = nothing, kwargs...)
    # TRAINH: optimal control for train on a hilly track
    # N=1001 time points, 4*(N+1)=4004 variables
    # Similar to TRAINF but with hill slope terms (atan elements) in VEQ

    TIME   = 4.8
    LENGTH = 6.0
    A_val  = 0.3
    B_val  = 0.14
    C_val  = 0.16
    UAMAX  = 10.0
    UBMIN  = -2.0
    EPS    = 0.05
    PI_val = 3.1415926535

    # Track sections: NS=3, breakpoints Z1=2, Z2=4, slopes S1=2, S2=0, S3=-2
    Z_bp = [2.0, 4.0]  # Z(1), Z(2)
    S_sl = [2.0, 0.0, -2.0]  # S(1), S(2), S(3)

    H    = TIME / N
    Hh   = H / 2
    BHh  = B_val * Hh
    AH   = A_val * H
    CHh  = C_val * Hh
    LperN = LENGTH / N
    H2PI = Hh / PI_val

    # DS(J) = S(J+1) - S(J) for J=1..NS-1=2
    # WJ(J) = DS(J) * H2PI
    W_vals = [(S_sl[J+1] - S_sl[J]) * H2PI for J in 1:2]
    CNST = -AH   # since SUMS = S1+S3 = 0

    NP1 = N + 1
    NV = 4 * NP1

    x_idx(I)  = I + 1
    v_idx(I)  = NP1 + I + 1
    ua_idx(I) = 2*NP1 + I + 1
    ub_idx(I) = 3*NP1 + I + 1

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    x0   = zeros(T, NV)

    # Boundary conditions
    lvar[x_idx(0)] = T(0);      uvar[x_idx(0)] = T(0);      x0[x_idx(0)] = T(0)
    lvar[v_idx(0)] = T(0);      uvar[v_idx(0)] = T(0);      x0[v_idx(0)] = T(0)
    lvar[ua_idx(0)] = T(UAMAX); uvar[ua_idx(0)] = T(UAMAX); x0[ua_idx(0)] = T(UAMAX)
    lvar[ub_idx(0)] = T(0);     uvar[ub_idx(0)] = T(0);     x0[ub_idx(0)] = T(0)
    lvar[x_idx(N)] = T(LENGTH); uvar[x_idx(N)] = T(LENGTH); x0[x_idx(N)] = T(LENGTH)
    lvar[v_idx(N)] = T(0);      uvar[v_idx(N)] = T(0);      x0[v_idx(N)] = T(0)
    lvar[ua_idx(N)] = T(0);     uvar[ua_idx(N)] = T(0);     x0[ua_idx(N)] = T(0)
    lvar[ub_idx(N)] = T(UBMIN); uvar[ub_idx(N)] = T(UBMIN); x0[ub_idx(N)] = T(UBMIN)

    for I in 1:N-1
        lvar[x_idx(I)]  = T(-Inf); uvar[x_idx(I)]  = T(Inf);  x0[x_idx(I)]  = T(LperN * I)
        lvar[v_idx(I)]  = T(-Inf); uvar[v_idx(I)]  = T(Inf);  x0[v_idx(I)]  = T(LperN)
        lvar[ua_idx(I)] = T(0);    uvar[ua_idx(I)] = T(UAMAX); x0[ua_idx(I)] = T(0)
        lvar[ub_idx(I)] = T(UBMIN);uvar[ub_idx(I)] = T(0);    x0[ub_idx(I)] = T(0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: ENERGY = H * sum_{I=1}^{N-1} UA(I)*V(I)
    en_data = [(uai = ua_idx(I), vi = v_idx(I)) for I in 1:N-1]
    ExaModels.@add_obj(c, T(H) * x[d.uai] * x[d.vi] for d in en_data)

    W1 = T(W_vals[1]); W2 = T(W_vals[2])
    Z1 = T(Z_bp[1]);   Z2 = T(Z_bp[2])
    EPS_T = T(EPS)

    # CUTEst order: I-outer (XEQ(I), VEQ(I) for each I=0..N-1)
    for I in 0:N-1
        xi = x_idx(I); xip1 = x_idx(I+1)
        vi = v_idx(I);  vip1 = v_idx(I+1)
        uai = ua_idx(I); uaip1 = ua_idx(I+1)
        ubi = ub_idx(I); ubip1 = ub_idx(I+1)
        ExaModels.@add_con(c, x[xip1] - x[xi] - T(Hh)*x[vip1] - T(Hh)*x[vi])
        ExaModels.@add_con(c,
            T(1+BHh)*x[vip1] + T(BHh-1)*x[vi]
            - T(Hh)*(x[uaip1] + x[uai] + x[ubip1] + x[ubi])
            + T(CHh)*(x[vi]^2 + x[vip1]^2) + T(CNST)
            + W1 * atan((x[xi]   - Z1) / EPS_T) + W1 * atan((x[xip1] - Z1) / EPS_T)
            + W2 * atan((x[xi]   - Z2) / EPS_T) + W2 * atan((x[xip1] - Z2) / EPS_T))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
