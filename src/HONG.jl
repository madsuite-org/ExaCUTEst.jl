# HONG: Chemical process optimization with exponential terms (Hong/Apte 1991)
# 4 variables: T1..T4 ∈ [0,1]; start = 0.5
# Objective: sum of 4 P3+P4*exp(P5*(P1+P2*X)) terms (each function of one variable)
# Equality constraint: T1 + T2 + T3 + T4 = 1
function hong_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T(0), uvar = T(1), start = T(0.5))
    ExaModels.@add_obj(c,
        T(0.92)    + T(0.08)     * exp(T(0.38)   * T(25)     * x[1])
        + T(-2.95) + T(3.95)     * exp(T(0.11)   * T(50)     * x[2])
        + T(-1.66) + T(1657834)  * exp(T(-1.48)  * (T(9) - T(4)*x[3]))
        + T(0.11)  + T(0.89)     * exp(T(0.00035)* T(20000)  * x[4]))
    ExaModels.@add_con(c, x[1] + x[2] + x[3] + x[4] - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
