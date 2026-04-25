# GROWTH: Gaussian elimination growth fitting (Gould 1991)
# 3 free variables: U1=100, U2=U3=0; 12 equality constraints
# G(n): U1 * n^(U2 + log(n)*U3) - target = 0
# Same element as GROWTHLS but E-type groups (no L2 group type applied)
function growth_model(; T = Float64, backend = nothing, kwargs...)
    ns   = T[8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25]
    ys   = T[8.0, 8.4305, 9.5294, 10.4627, 12.0, 13.0205, 14.5949,
             16.1078, 18.0596, 20.4569, 24.25, 32.9863]
    logn = log.(ns)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(100), T(0), T(0)])
    ExaModels.@add_par(c, LOGNI, logn)
    ExaModels.@add_par(c, YI, ys)
    ExaModels.@add_con(c, x[1] * exp(LOGNI[i] * (x[2] + LOGNI[i] * x[3])) - YI[i]
        for i in 1:12)
    return ExaModels.ExaModel(c; kwargs...)
end
