function camshape_model(N = 800; T = Float64, backend = nothing, kwargs...)
    rmin    = T(1)
    rmax    = T(2)
    rav     = (rmin + rmax) / 2
    alpha   = T(1.5)
    dtheta  = T(2) * T(π) / T(5 * (N + 1))
    cdtheta = cos(dtheta)
    adtheta = alpha * dtheta
    pirv_n  = T(π) / T(N)        # PI * RV / N with RV=1

    rmin2   = rmin^2
    rmin2cd = rmin * 2 * cdtheta
    rmax2cd = rmax * 2 * cdtheta

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(rmin, N), uvar = fill(rmax, N), start = fill(rav, N))

    # Objective: minimize -(PI*RV/N)*sum R(i)
    ExaModels.@add_obj(c, -pirv_n * x[i] for i in 1:N)

    # Convexity CO(I) for I=2..N-1 (L type, ≤ 0)
    # -R(I-1)*R(I) - R(I)*R(I+1) + 2*CDTHETA*R(I-1)*R(I+1) ≤ 0
    co_data = [(im1 = i - 1, i = i, ip1 = i + 1) for i in 2:N-1]
    ExaModels.@add_con(c,
        -(x[d.im1] * x[d.i]) - (x[d.ip1] * x[d.i]) + T(2) * cdtheta * (x[d.im1] * x[d.ip1])
        for d in co_data; lcon = T(-Inf), ucon = T(0))

    # E1: -RMIN*R(1) + RMIN*2CDTHETA*R(2) - R(1)*R(2) ≤ 0
    ExaModels.@add_con(c, -rmin * x[1] + rmin2cd * x[2] - x[1] * x[2]; lcon = T(-Inf), ucon = T(0))

    # E2: RMIN*(2CDTHETA-1)*R(1) - RMIN^2 ≤ 0
    ExaModels.@add_con(c, rmin * (T(2) * cdtheta - T(1)) * x[1] - rmin2; lcon = T(-Inf), ucon = T(0))

    # E3: -RMAX*R(N) + RMAX*2CDTHETA*R(N-1) - R(N)*R(N-1) ≤ 0
    ExaModels.@add_con(c, -rmax * x[N] + rmax2cd * x[N-1] - x[N] * x[N-1]; lcon = T(-Inf), ucon = T(0))

    # E4: -2*RMAX*R(N) + 2*CDTHETA*R(N)^2 ≤ 0
    ExaModels.@add_con(c, T(-2) * rmax * x[N] + T(2) * cdtheta * x[N]^2; lcon = T(-Inf), ucon = T(0))

    # CU(0): R(1) - RMIN + ADTHETA, lcon=0, ucon=2*ADTHETA
    ExaModels.@add_con(c, x[1] - rmin + adtheta; lcon = T(0), ucon = T(2) * adtheta)

    # CU(I) for I=1..N-1: R(I+1) - R(I) + ADTHETA, lcon=0, ucon=2*ADTHETA
    cu_data = [(i = i, ip1 = i + 1) for i in 1:N-1]
    ExaModels.@add_con(c, x[d.ip1] - x[d.i] + adtheta for d in cu_data; lcon = T(0), ucon = T(2) * adtheta)

    # CU(N): -R(N) + RMAX + ADTHETA, lcon=0, ucon=2*ADTHETA
    ExaModels.@add_con(c, -x[N] + rmax + adtheta; lcon = T(0), ucon = T(2) * adtheta)

    return ExaModels.ExaModel(c; kwargs...)
end
