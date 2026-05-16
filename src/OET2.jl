# OET2: Nonlinear Chebychev problem
# phi(x,w) = 1/(1+w) - x1*exp(w*x2)
# min u  s.t.  u - phi >= 0, u + phi >= 0 for w in discretized [-0.5, 0.5]
# Variables: u(1), x1(2), x2(3); M=500 discrete points
function oet2_model(M = 500; T = Float64, backend = nothing, kwargs...)
    LOWER = T(-0.5)
    UPPER = T(0.5)
    H = (UPPER - LOWER) / T(M)

    # Precompute w values and RHS constants
    W = T[LOWER + T(i) * H for i in 0:M]
    rhs_lo = T[-1 / (1 + W[i+1]) for i in 0:M]   # -1/(1+w)
    rhs_up = T[ 1 / (1 + W[i+1]) for i in 0:M]   #  1/(1+w)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # Variables: U(1), X1(2), X2(3) — all free
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf), start = T(0.0))

    # Objective: min U
    ExaModels.@add_obj(c, x[1])

    # Constraints LO(I): U - x1*exp(w*x2) >= -1/(1+w)
    # i.e., U - x1*exp(w*x2) - (-1/(1+w)) >= 0  → lcon=0, ucon=Inf
    ExaModels.@add_par(c, wi, W)
    ExaModels.@add_par(c, rlo, rhs_lo)
    ExaModels.@add_par(c, rup, rhs_up)
    ExaModels.@add_con(c, x[1] - x[2] * exp(wi[i+1] * x[3]) - rlo[i+1] for i in 0:M;
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[1] + x[2] * exp(wi[i+1] * x[3]) - rup[i+1] for i in 0:M;
                       lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
