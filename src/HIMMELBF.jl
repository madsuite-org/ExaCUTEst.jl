# HIMMELBF: Himmelblau data fitting problem (4 variables, unconstrained)
# Source: Himmelblau (1972), problem 32; Buckley#76
# 7 L2 groups G(i), sigma=0.0001, CONSTANT=1.0
# Element HF: F = (x1^2 + A(i)*x2^2 + A(i)^2*x3^2) / (B(i)*(1 + A(i)*x4^2))
# contribution = (F - 1)^2 / 0.0001
function himmelbf_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # All variables free (FR DEFAULT)
    ExaModels.@add_var(c, x, 4; lvar = T[-Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf],
                       start = T[2.7, 90.0, 1500.0, 10.0])
    A = T[0.0, 0.000428, 0.001000, 0.001610, 0.002090, 0.003480, 0.005250]
    B = T[7.391, 11.18, 16.44, 16.20, 22.20, 24.02, 31.32]
    ExaModels.@add_par(c, Ai, A)
    ExaModels.@add_par(c, Bi, B)
    # v_i = (x1^2 + Ai*x2^2 + Ai^2*x3^2) / (Bi*(1 + Ai*x4^2)) - 1.0
    # contribution = v_i^2 / 0.0001
    ExaModels.@add_obj(c,
        ((x[1]^2 + Ai[i]*x[2]^2 + Ai[i]^2*x[3]^2) / (Bi[i]*(T(1) + Ai[i]*x[4]^2)) - T(1))^2 / T(0.0001)
        for i in 1:7)
    return ExaModels.ExaModel(c; kwargs...)
end
