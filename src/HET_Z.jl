# HET-Z: Nonlinear Chebychev approximation
# min u  s.t. u - phi(x,w) >= 0, u + phi(x,w) >= 0  for w in discretized [-1,1]
# phi(x,w) = (1-w^2) - (0.5*x^2 - 2*x*w)
# Element QUAD: 0.5*x^2 (with coeff -1 → -0.5*x^2)
# LO(i): u + 2w*x - QUAD >= rhs(i) where rhs = -1+w^2  → u + 2w*x - 0.5*x^2 - (-1+w^2) >= 0
#       i.e. u + 2w*x - 0.5*x^2 + 1 - w^2 >= 0
# UP(i): u - 2w*x + QUAD >= rhs(i) where rhs = 1-w^2 → u - 2w*x + 0.5*x^2 - (1-w^2) >= 0
#       i.e. u - 2w*x + 0.5*x^2 - 1 + w^2 >= 0
# M = 500, step H = (1-(-1))/500 = 0.002, w_i = -1 + i*H for i=0..500
function het_z_model(M = 500; T = Float64, backend = nothing, kwargs...)
    H = T(2) / M  # (upper - lower) / M = 2/500
    lower = T(-1)
    ws = T[lower + i * H for i in 0:M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # Variables: U (free), X (free)
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf))

    # LO(i): u + 2*w*x - 0.5*x^2 - (-1+w^2) >= 0
    #        u + 2*w*x - 0.5*x^2 + 1 - w^2 >= 0
    # Constraint: x[1] + 2*ws[i]*x[2] - 0.5*x[2]^2 + 1 - ws[i]^2 - rhs = 0
    # with lcon=0, ucon=Inf → G type
    ExaModels.@add_par(c, WLO, ws)
    ExaModels.@add_con(c,
        x[1] + T(2) * WLO[i] * x[2] - T(0.5) * x[2]^2 + T(1) - WLO[i]^2
        for i in 1:(M+1);
        lcon = T(0), ucon = T(Inf))

    # UP(i): u - 2*w*x + 0.5*x^2 - (1-w^2) >= 0
    #        u - 2*w*x + 0.5*x^2 - 1 + w^2 >= 0
    ExaModels.@add_par(c, WUP, ws)
    ExaModels.@add_con(c,
        x[1] - T(2) * WUP[i] * x[2] + T(0.5) * x[2]^2 - T(1) + WUP[i]^2
        for i in 1:(M+1);
        lcon = T(0), ucon = T(Inf))

    ExaModels.@add_obj(c, x[1])

    return ExaModels.ExaModel(c; kwargs...)
end
