function tfi1_model(M = 100; T = Float64, backend = nothing, kwargs...)
    # TFI1: Semi-infinite minmax, discretized to M+1 constraints
    # Variables: X1, X2, X3 (all free: FR DEFAULT)
    # Objective: X1^2 + X2^2 + X3^2
    # Constraints CG(I) for I=0..M (L≤0):
    #   X1 + X2*exp(X3*T_I) ≤ 2*sin(4*T_I) - exp(2*T_I)
    #   i.e., X1 + X2*exp(X3*T_I) - (2*sin(4*T_I) - exp(2*T_I)) ≤ 0
    # where T_I = I/M, H = 1/M

    H = 1 / M
    cg_data = [(i = I, t = I * H, rhs = 2*sin(4*I*H) - exp(2*I*H)) for I in 0:M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[-Inf, -Inf, -Inf],
        uvar = T[Inf, Inf, Inf],
        start = T[1.0, 1.0, 1.0])

    # Objective: X1^2 + X2^2 + X3^2
    ExaModels.@add_obj(c, x[1]^2 + x[2]^2 + x[3]^2)

    # Constraints CG(I) (L≤0): X1 + X2*exp(X3*t) - rhs ≤ 0
    ExaModels.@add_par(c, tvals, T[d.t for d in cg_data])
    ExaModels.@add_par(c, rhs, T[d.rhs for d in cg_data])
    ExaModels.@add_con(c,
        x[1] + x[2] * exp(x[3] * tvals[i]) - rhs[i]
        for i in 1:(M+1); lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
