function tfi3_model(M = 100; T = Float64, backend = nothing, kwargs...)
    # TFI3: Semi-infinite minmax, discretized to M+1 constraints
    # Variables: X1, X2, X3 (all free: FR DEFAULT)
    # Objective: exp(X1) + exp(X2) + exp(X3)
    # Constraints CG(I) for I=0..M (L≤0):
    #   -X1 - T*X2 - T^2*X3 + 1/(1+T^2) ≤ 0
    #   (RHS = -1/(1+T^2) = -1/1+TT, linear = -X1 - T*X2 - T^2*X3)
    # where T_I = I/M, H = 1/M
    # Start: X1=1, X2=0.5, X3=0

    H = 1 / M
    cg_data = [(t = I * H, tt = (I * H)^2, rhs_val = 1/(1+(I*H)^2)) for I in 0:M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[-Inf, -Inf, -Inf],
        uvar = T[Inf, Inf, Inf],
        start = T[1.0, 0.5, 0.0])

    # Objective: exp(X1) + exp(X2) + exp(X3)
    ExaModels.@add_obj(c, exp(x[1]) + exp(x[2]) + exp(x[3]))

    # Constraints CG(I) (L≤0): -X1 - t*X2 - tt*X3 + 1/(1+tt) ≤ 0
    ExaModels.@add_par(c, tvals, T[d.t for d in cg_data])
    ExaModels.@add_par(c, ttvals, T[d.tt for d in cg_data])
    ExaModels.@add_par(c, rhs_par, T[d.rhs_val for d in cg_data])
    ExaModels.@add_con(c,
        -x[1] - tvals[i]*x[2] - ttvals[i]*x[3] + rhs_par[i]
        for i in 1:(M+1); lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
