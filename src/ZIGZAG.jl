function zigzag_model(T_steps = 500; T = Float64, backend = nothing, kwargs...)
    # ZIGZAG: minimum-time control on sine trajectory, T_steps=500
    # 3004 variables: X(0..T), Y(0..T), VX(0..T), VY(0..T) [fixed at boundaries],
    #                 UX(1..T), UY(1..T)
    # Variable indexing (1-based):
    #   X(I)  = I+1 for I=0..T        [1..T+1 = 1..501]
    #   Y(I)  = T+2+I for I=0..T      [502..1002]
    #   VX(I) = 2*(T+1)+I+1 for I=0..T [1003..1503]
    #   VY(I) = 3*(T+1)+I+1 for I=0..T [1504..2004]
    #   UX(I) = 4*(T+1)+I for I=1..T  [2005..2504]... wait
    # Let me recount carefully.

    Ts = T_steps
    XT  = 10.0
    MASS = 0.5
    TOL  = 0.1
    H    = XT / Ts
    MoH  = MASS / H   # M/H
    FoH  = 1 / H      # 1/H
    FMAX = XT / Ts    # = H
    W    = XT * (Ts + 1) / 2  # = XT*(T+1)/2

    NP1 = Ts + 1  # T+1 points

    # Variable layout (1-indexed):
    x_idx(I)  = I + 1          # I=0..T → 1..NP1
    y_idx(I)  = NP1 + I + 1    # I=0..T → NP1+1..2*NP1
    vx_idx(I) = 2*NP1 + I + 1  # I=0..T → 2NP1+1..3NP1
    vy_idx(I) = 3*NP1 + I + 1  # I=0..T → 3NP1+1..4NP1
    ux_idx(I) = 4*NP1 + I      # I=1..T → 4NP1+1..4NP1+T
    uy_idx(I) = 4*NP1 + Ts + I # I=1..T → 4NP1+T+1..4NP1+2T

    NV = 4*NP1 + 2*Ts  # = 4*(T+1) + 2T = 6T+4

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    x0   = zeros(T, NV)

    # Fixed boundary conditions
    for var_idx in [x_idx(0), y_idx(0), vx_idx(0), vy_idx(0)]
        lvar[var_idx] = T(0); uvar[var_idx] = T(0); x0[var_idx] = T(0)
    end
    lvar[vx_idx(Ts)] = T(0); uvar[vx_idx(Ts)] = T(0)
    lvar[vy_idx(Ts)] = T(0); uvar[vy_idx(Ts)] = T(0)

    # X(I) ∈ [0, XT] for I=1..T
    for I in 1:Ts
        lvar[x_idx(I)] = T(0); uvar[x_idx(I)] = T(XT); x0[x_idx(I)] = T(I*H)
        x0[vx_idx(I)] = T(1)
    end

    # UX(I), UY(I) ∈ [-FMAX, FMAX] for I=1..T
    for I in 1:Ts
        lvar[ux_idx(I)] = T(-FMAX); uvar[ux_idx(I)] = T(FMAX)
        lvar[uy_idx(I)] = T(-FMAX); uvar[uy_idx(I)] = T(FMAX)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: sum_{I=1}^T (X(I) - XT)^2 / (W/T_I) where T_I = I*H
    # = sum (X(I)-XT)^2 * I*H / W = sum (X(I)-XT)^2 * I / (W/H)
    ExaModels.@add_par(c, obj_scale, T[T(W / (I * H)) for I in 1:Ts])
    obj_data = [(I = I, xi = x_idx(I)) for I in 1:Ts]
    ExaModels.@add_obj(c,
        (x[d.xi] - T(XT))^2 / obj_scale[d.I]
        for d in obj_data)

    # Constraints:
    # ACX(I): MoH*VX(I) - MoH*VX(I-1) - UX(I) = 0
    acx_data = [(vxi = vx_idx(I), vxim1 = vx_idx(I-1), uxi = ux_idx(I)) for I in 1:Ts]
    ExaModels.@add_con(c,
        T(MoH)*x[d.vxi] - T(MoH)*x[d.vxim1] - x[d.uxi]
        for d in acx_data)

    # ACY(I): MoH*VY(I) - MoH*VY(I-1) - UY(I) = 0
    acy_data = [(vyi = vy_idx(I), vyim1 = vy_idx(I-1), uyi = uy_idx(I)) for I in 1:Ts]
    ExaModels.@add_con(c,
        T(MoH)*x[d.vyi] - T(MoH)*x[d.vyim1] - x[d.uyi]
        for d in acy_data)

    # PSX(I): FoH*X(I) - FoH*X(I-1) - VX(I) = 0
    psx_data = [(xi = x_idx(I), xim1 = x_idx(I-1), vxi = vx_idx(I)) for I in 1:Ts]
    ExaModels.@add_con(c,
        T(FoH)*x[d.xi] - T(FoH)*x[d.xim1] - x[d.vxi]
        for d in psx_data)

    # PSY(I): FoH*Y(I) - FoH*Y(I-1) - VY(I) = 0
    psy_data = [(yi = y_idx(I), yim1 = y_idx(I-1), vyi = vy_idx(I)) for I in 1:Ts]
    ExaModels.@add_con(c,
        T(FoH)*x[d.yi] - T(FoH)*x[d.yim1] - x[d.vyi]
        for d in psy_data)

    # SC(I): Y(I) - sin(X(I)) + TOL ∈ [0, 2*TOL]
    sc_data = [(yi = y_idx(I), xi = x_idx(I)) for I in 1:Ts]
    ExaModels.@add_con(c,
        x[d.yi] - sin(x[d.xi]) + T(TOL)
        for d in sc_data; lcon = T(0), ucon = T(2*TOL))

    return ExaModels.ExaModel(c; kwargs...)
end
