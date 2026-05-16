function vibrbeam_model(; T = Float64, backend = nothing, kwargs...)
    # VIBRBEAM: vibrating beam fit, 8 variables, 30 residuals
    # v_i ≈ (c0 + c1*x + c2*x^2 + c3*x^3) * cos(d0 + x*(d1+x*(d2+x*d3)) - p)
    # Objective: sum_{i=1}^30 (above - v_i)^2
    # All 8 variables free (XR = free)
    # Start: c0=-3.5, c1=1.0, d0=1.7; rest=0

    x_data = T[39.1722, 53.9707, 47.9829, 12.5925, 16.5414, 18.9548, 27.7168, 31.9201,
               45.6830, 22.2524, 33.9805, 6.8425, 35.1677, 33.5682, 43.3659, 13.3835,
               25.7273, 21.0230, 10.9755, 1.5323, 45.4416, 14.5431, 22.4313, 29.0144,
               25.2675, 15.5095, 9.6297, 8.3009, 30.8694, 43.3299]

    v_data = T[-1.2026, 1.7053, 0.5410, 1.1477, 1.2447, 0.9428, -0.1360, -0.7542,
               -0.3396, 0.7057, -0.8509, -0.1201, -1.2193, -1.0448, -0.7723, 0.4342,
               0.1154, 0.2868, 0.3558, -0.5090, -0.0842, 0.6021, 0.1197, -0.1827,
               0.1806, 0.5395, 0.2072, 0.1466, -0.2672, -0.3038]

    p_data = T[2.5736, 2.7078, 2.6613, 2.0374, 2.1553, 2.2195, 2.4077, 2.4772,
               2.6409, 2.2981, 2.5073, 1.8380, 2.5236, 2.5015, 2.6186, 0.4947,
               0.6062, 0.5588, 0.4772, 0.4184, 0.9051, 0.5035, 0.5723, 0.6437,
               0.6013, 0.5111, 0.4679, 0.4590, 0.6666, 0.8630]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # Variables: c0,c1,c2,c3,d0,d1,d2,d3 (all free: XR)
    ExaModels.@add_var(c, x, 8;
        lvar = T(-Inf), uvar = T(Inf),
        start = T[-3.5, 1.0, 0.0, 0.0, 1.7, 0.0, 0.0, 0.0])

    # x[1]=c0, x[2]=c1, x[3]=c2, x[4]=c3
    # x[5]=d0, x[6]=d1, x[7]=d2, x[8]=d3
    ExaModels.@add_par(c, xp, x_data)
    ExaModels.@add_par(c, vp, v_data)
    ExaModels.@add_par(c, pp, p_data)

    ExaModels.@add_obj(c,
        begin
            yi = xp[i]
            phi = x[5] + yi*(x[6] + yi*(x[7] + yi*x[8])) - pp[i]
            mag = x[1] + yi*(x[2] + yi*(x[3] + yi*x[4]))
            (mag * cos(phi) - vp[i])^2
        end
        for i in 1:30)

    return ExaModels.ExaModel(c; kwargs...)
end
