# FLETBV3M: Boundary value problem variant (Luksan et al), N=5000, unconstrained
# Variables: x(1..N), free, start x(I) = I*H where H=1/(N+1)
# Objective = (P/2)*[x1^2 + sum_{I=1}^{N-1}(xI-xI+1)^2 + xN^2]
#           + sum_I (-kappa/H^2)*P*cos(x(I))
#           + sum_I P*(-1-2/H^2)*100*sin(0.01*x(I))
# P = 1/1e8, H = 1/(N+1), kappa = 1
function fletbv3m_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    objscale = T(1e8)
    P_val    = T(1) / objscale
    H        = T(1) / T(N + 1)
    H2       = H * H
    kappa    = T(1)
    kappa_H2 = kappa / H2
    inv_1_2H2 = T(1) + T(2) / H2  # 1 + 2/H^2

    x0 = [T(i) * H for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # G(0): 0.5*P*x[1]^2
    ExaModels.@add_obj(c, T(0.5) * P_val * x[1]^2)

    # G(I) for I=1..N-1: 0.5*P*(x[I]-x[I+1])^2
    ExaModels.@add_obj(c,
        T(0.5) * P_val * (x[i] - x[i + 1])^2
        for i in 1:(N - 1))

    # G(N): 0.5*P*x[N]^2
    ExaModels.@add_obj(c, T(0.5) * P_val * x[N]^2)

    # C(I): (-kappa/H^2)*P*cos(x(I))
    ExaModels.@add_obj(c,
        (-kappa_H2) * P_val * cos(x[i])
        for i in 1:N)

    # S group: P*-1-2/H2 = (1+2/H^2)*P (positive, name misleading), SIN element = 100*sin(0.01*V)
    ExaModels.@add_obj(c,
        P_val * inv_1_2H2 * T(100) * sin(T(0.01) * x[i])
        for i in 1:N)

    return ExaModels.ExaModel(c; kwargs...)
end
