# FLETCHCR: Chained Rosenbrock function (Fletcher, NA/145, Dundee 1992)
# N=1000 variables, all free, start=0.0
# Objective = sum_{I=1}^{N-1} [100*(X(I+1)-X(I)^2)^2 + (1-X(I))^2]
# (standard chained Rosenbrock: L2, scale=0.01 for first term)
function fletchcr_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0))
    ExaModels.@add_obj(c, T(100) * (x[i+1] - x[i]^2)^2 for i in 1:N-1)
    ExaModels.@add_obj(c, (T(1) - x[i])^2 for i in 1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
