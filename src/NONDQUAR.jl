# NONDQUAR: nondiagonal quartic test problem (N=5000 variables, unconstrained)
# minimize sum_{i=1}^{N-2} (x[i]+x[i+1]+x[N])^4 + (x[1]-x[2])^2 + (x[N-1]-x[N])^2
# Source: Conn, Gould, Lescrenier, Toint (1988), problem 57
# SIF: L(I) groups with L4 type (v^4) for i=1..N-2; L(N-1) and L(N) with L2 type (v^2)
# All variables FREE; start: x[odd]=1.0, x[even]=-1.0 (SIF DI I 2 loop)
function nondquar_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    start_vec = T[isodd(i) ? 1.0 : -1.0 for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start_vec)
    ExaModels.@add_obj(c, (x[i] + x[i+1] + x[N])^4 for i in 1:N-2)
    ExaModels.@add_obj(c, (x[1] - x[2])^2)
    ExaModels.@add_obj(c, (x[N-1] - x[N])^2)
    return ExaModels.ExaModel(c; kwargs...)
end
