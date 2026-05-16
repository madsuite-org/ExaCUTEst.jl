# OET7: Nonlinear Chebychev problem
# phi(x,w) = 1/(1+w) - x1*exp(w*x4) - x2*exp(w*x5) - x3*exp(w*x6)
# Variables: u(1), x1(2), x2(3), x3(4), x4(5), x5(6), x6(7); M=500
function oet7_model(M = 500; T = Float64, backend = nothing, kwargs...)
    LOWER = T(-0.5)
    UPPER = T(0.5)
    H = (UPPER - LOWER) / T(M)

    W = T[LOWER + T(i) * H for i in 0:M]
    rhs = T[1 / (1 + W[i+1]) for i in 0:M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 7; lvar = T(-Inf), uvar = T(Inf), start = T(0.0))

    ExaModels.@add_obj(c, x[1])

    ExaModels.@add_par(c, wi, W)
    ExaModels.@add_par(c, rh, rhs)

    ExaModels.@add_con(c,
        x[1] - x[2]*exp(wi[i+1]*x[5]) - x[3]*exp(wi[i+1]*x[6]) - x[4]*exp(wi[i+1]*x[7]) + rh[i+1]
        for i in 0:M; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c,
        x[1] + x[2]*exp(wi[i+1]*x[5]) + x[3]*exp(wi[i+1]*x[6]) + x[4]*exp(wi[i+1]*x[7]) - rh[i+1]
        for i in 0:M; lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
