# OET5: Nonlinear Chebychev problem
# phi(x,w) = sqrt(w) - x4 - (x1*w^2 + x2*w + x3)^2
# min u  s.t.  LO(I): u + phi >= 0 (i.e., u - QUAD + x4 >= sqrt(w))
#              UP(I): u - phi >= 0 (i.e., u + QUAD - x4 >= -sqrt(w))
# Variables: u(1), x1(2), x2(3), x3(4), x4(5); M=500
function oet5_model(M = 500; T = Float64, backend = nothing, kwargs...)
    LOWER = T(0.25)
    UPPER = T(1.0)
    H = (UPPER - LOWER) / T(M)

    W = T[LOWER + T(i) * H for i in 0:M]
    sqrtW = T[sqrt(W[i+1]) for i in 0:M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # All variables free; start at 1.0
    ExaModels.@add_var(c, x, 5; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))

    ExaModels.@add_obj(c, x[1])

    ExaModels.@add_par(c, wi, W)
    ExaModels.@add_par(c, sw, sqrtW)

    # LO(I): u + x4*(-1) + QUAD*1 >= sqrt(w)
    # In SIF: XG LO(I) U 1.0  X4 -1.0; CONSTANT -sqrt(w); element QUAD coeff 1
    # => U - x4 + (x1*w^2+x2*w+x3)^2 - (-sqrt(w)) >= 0
    # => U - x4 + (x1*w^2+x2*w+x3)^2 + sqrt(w) >= 0, lcon=0
    ExaModels.@add_con(c, x[1] - x[5] + (x[2]*wi[i+1]^2 + x[3]*wi[i+1] + x[4])^2 + sw[i+1]
                       for i in 0:M; lcon = T(0), ucon = T(Inf))
    # UP(I): u + x4*1 + QUAD*(-1) >= sqrt(w)
    # => U + x4 - (x1*w^2+x2*w+x3)^2 - sqrt(w) >= 0
    ExaModels.@add_con(c, x[1] + x[5] - (x[2]*wi[i+1]^2 + x[3]*wi[i+1] + x[4])^2 - sw[i+1]
                       for i in 0:M; lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
