# DQRTIC: Variable dimension diagonal quartic problem
# N=5000 variables, all free, start=2.0
# GROUP TYPE L4: F(v)=v^4; G(I): v=x[I]-I → contribution=(x[I]-I)^4
function dqrtic_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(2))
    ExaModels.@add_par(c, RHS, T[T(i) for i in 1:N])
    ExaModels.@add_obj(c, (x[i] - RHS[i])^4 for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
