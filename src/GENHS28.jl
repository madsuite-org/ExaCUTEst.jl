# GENHS28: Variable-size generalization of HS28
# N=10 variables, all free, start: x[1]=-4.0, rest=1.0
# Objective: sum_{I=1}^{N-1} (X(I)+X(I+1))^2
# Constraints: X(I) + 2*X(I+1) + 3*X(I+2) = 1  for I=1..N-2
function genhs28_model(N = 10; T = Float64, backend = nothing, kwargs...)
    start = [T(-4); fill(T(1), N - 1)]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    ExaModels.@add_obj(c, (x[i] + x[i+1])^2 for i in 1:N-1)
    ExaModels.@add_con(c, x[i] + T(2)*x[i+1] + T(3)*x[i+2] - T(1) for i in 1:N-2)
    return ExaModels.ExaModel(c; kwargs...)
end
