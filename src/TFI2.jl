function tfi2_model(M = 100; T = Float64, backend = nothing, kwargs...)
    # TFI2: Semi-infinite minmax, discretized to M+1 constraints
    # Variables: X1, X2, X3 (all free: FR DEFAULT)
    # Objective: X1 + 0.5*X2 + X3/3
    # Constraints CG(I) for I=0..M (L≤0):
    #   -X1 - T*X2 - T^2*X3 + tan(T) ≤ 0
    #   (RHS = -tan(T), linear = -X1 - T*X2 - T^2*X3,
    #    v = linear - rhs = -X1 - T*X2 - T^2*X3 + tan(T) ≤ 0)
    # where T_I = I/M, H = 1/M
    # No START POINT section → default start (0, 0, 0)

    H = 1 / M
    cg_data = [(t = I * H, tt = (I * H)^2, tant = tan(I * H)) for I in 0:M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[-Inf, -Inf, -Inf],
        uvar = T[Inf, Inf, Inf],
        start = T[0.0, 0.0, 0.0])

    # Objective: X1 + 0.5*X2 + X3/3
    ExaModels.@add_obj(c, x[1] + T(0.5)*x[2] + x[3]/T(3))

    # Constraints CG(I) (L≤0): -X1 - t*X2 - tt*X3 + tant ≤ 0
    ExaModels.@add_par(c, tvals, T[d.t for d in cg_data])
    ExaModels.@add_par(c, ttvals, T[d.tt for d in cg_data])
    ExaModels.@add_par(c, tantvals, T[d.tant for d in cg_data])
    ExaModels.@add_con(c,
        -x[1] - tvals[i]*x[2] - ttvals[i]*x[3] + tantvals[i]
        for i in 1:(M+1); lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
