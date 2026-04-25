# SIPOW4: Semi-infinite LP, circle boundary approximation (Powell 1992)
# 4 free variables: X1..X4; start = [-0.1, 0, 0, 1.2]
# Objective: X4; M/2 G-type + M/2 L-type constraints
# Points (xi,eta) parametrize the circle (xi-0.5)²+(eta-0.5)²=0.5
# xi(j)  = 0.5 - cos(π/4 - j*2π/M) * √0.5
# eta(j) = 0.5 - sin(π/4 - j*2π/M) * √0.5
function sipow4_model(M = 2000; T = Float64, backend = nothing, kwargs...)
    M2 = M ÷ 2
    twopi_M = 2π / M
    roothalf = sqrt(0.5)
    data = let
        [(xi  = T(0.5 - cos(π/4 - j*twopi_M) * roothalf),
          eta = T(0.5 - sin(π/4 - j*twopi_M) * roothalf),
          rhs = T((0.5 - cos(π/4 - j*twopi_M)*roothalf)^2 *
                  (0.5 - sin(π/4 - j*twopi_M)*roothalf)))
         for j in 1:M2]
    end
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(-0.1), T(0), T(0), T(1.2)])
    ExaModels.@add_obj(c, x[4])
    ExaModels.@add_con(c, x[1] + d.xi*x[2] + d.eta*x[3] + x[4] - d.rhs
        for d in data; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[1] + d.xi*x[2] + d.eta*x[3] - d.rhs
        for d in data; lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
