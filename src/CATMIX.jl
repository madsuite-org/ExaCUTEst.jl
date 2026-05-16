function catmix_model(NH = 800; T = Float64, backend = nothing, kwargs...)
    # Variables declared DO I=0..NH: U(I), X1(I), X2(I)
    # 1-indexed: u(I) = 3I+1, x1(I) = 3I+2, x2(I) = 3I+3
    NV   = 3 * (NH + 1)
    tf   = T(1)
    h    = tf / T(NH)
    hh   = h / 2
    alpha = T(0)      # smoothing parameter (default 0)
    alphah = alpha * h

    u_idx(I)  = 3 * I + 1
    x1_idx(I) = 3 * I + 2
    x2_idx(I) = 3 * I + 3

    # Bounds: all free except U(I) ∈ [0,1] and X1(0), X2(0) fixed
    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    for I in 0:NH
        lvar[u_idx(I)] = T(0)
        uvar[u_idx(I)] = T(1)
    end
    lvar[x1_idx(0)] = T(1); uvar[x1_idx(0)] = T(1)   # X1(0) = 1 fixed
    lvar[x2_idx(0)] = T(0); uvar[x2_idx(0)] = T(0)   # X2(0) = 0 fixed

    x0 = zeros(T, NV)
    for I in 0:NH
        x0[u_idx(I)]  = T(0)
        x0[x1_idx(I)] = T(1)
        x0[x2_idx(I)] = T(0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: X1(NH) + X2(NH) - 1  (+ alpha*h*sum (U(i+1)-U(i))^2, but alpha=0)
    ExaModels.@add_obj(c, x[x1_idx(NH)] + x[x2_idx(NH)] - T(1))

    # ODE1(I) for I=0..NH-1: X1(I) - X1(I+1) + (H/2)*(U(I)*(10*X2(I)-X1(I)) + U(I+1)*(10*X2(I+1)-X1(I+1))) = 0
    ode_data = [(
        uI   = u_idx(I),   uI1  = u_idx(I + 1),
        x1I  = x1_idx(I),  x1I1 = x1_idx(I + 1),
        x2I  = x2_idx(I),  x2I1 = x2_idx(I + 1),
    ) for I in 0:NH-1]

    ExaModels.@add_con(c,
        x[d.x1I] - x[d.x1I1]
        + hh * (x[d.uI]  * (T(10) * x[d.x2I]  - x[d.x1I]))
        + hh * (x[d.uI1] * (T(10) * x[d.x2I1] - x[d.x1I1]))
        for d in ode_data)

    # ODE2(I) for I=0..NH-1:
    # X2(I) - X2(I+1) + (H/2)*(-U(I)*(10*X2(I)-X1(I)) + (U(I)-1)*X2(I))
    #                 + (H/2)*(-U(I+1)*(10*X2(I+1)-X1(I+1)) + (U(I+1)-1)*X2(I+1)) = 0
    ExaModels.@add_con(c,
        x[d.x2I] - x[d.x2I1]
        + hh * (-x[d.uI]  * (T(10) * x[d.x2I]  - x[d.x1I])  + (x[d.uI]  - T(1)) * x[d.x2I])
        + hh * (-x[d.uI1] * (T(10) * x[d.x2I1] - x[d.x1I1]) + (x[d.uI1] - T(1)) * x[d.x2I1])
        for d in ode_data)

    return ExaModels.ExaModel(c; kwargs...)
end
