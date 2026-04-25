# QUARTC: Quartic function (N=5000 variables, unconstrained)
# minimize sum_{i=1}^{N} (x[i] - i)^4
# Source: Buckley (1989), problem 157
# SIF: G(I) groups, linear x[I]*1.0, constant REALI=i (RHS), L4 group type F(v)=v^4
#   v = x[I] - i, contribution = (x[I] - i)^4
#   Bounds: FR DEFAULT → all variables free
#   Start: all x[i] = 2.0
function quartc_model(; T = Float64, backend = nothing, kwargs...)
    N = 5000
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(2))
    ExaModels.@add_par(c, θ, T[T(i) for i in 1:N])
    ExaModels.@add_obj(c, (x[i] - θ[i])^4 for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
