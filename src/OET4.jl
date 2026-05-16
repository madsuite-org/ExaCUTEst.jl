# OET4: Nonlinear Chebychev problem
# phi(x,w) = exp(w) - (x1 + x2*w) / (1 + x3*w)
# min u  s.t.  u - phi >= 0, u + phi >= 0 for w in [0, 1]
# Variables: u(1), x1(2), x2(3), x3(4); M=500
function oet4_model(M = 500; T = Float64, backend = nothing, kwargs...)
    LOWER = T(0.0)
    UPPER = T(1.0)
    H = (UPPER - LOWER) / T(M)

    W = T[LOWER + T(i) * H for i in 0:M]
    expW = T[exp(W[i+1]) for i in 0:M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T(-Inf), uvar = T(Inf), start = T(0.0))

    ExaModels.@add_obj(c, x[1])

    ExaModels.@add_par(c, wi, W)
    ExaModels.@add_par(c, ew, expW)

    # LO(I): u - RATIO >= -exp(w), i.e. u - (x1+x2*w)/(1+x3*w) + exp(w) >= 0
    # UP(I): u + RATIO >= exp(w), i.e. u + (x1+x2*w)/(1+x3*w) - exp(w) >= 0
    ExaModels.@add_con(c, x[1] - (x[2] + x[3] * wi[i+1]) / (T(1) + x[4] * wi[i+1]) + ew[i+1]
                       for i in 0:M; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[1] + (x[2] + x[3] * wi[i+1]) / (T(1) + x[4] * wi[i+1]) - ew[i+1]
                       for i in 0:M; lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
