# BROWNAL: Brown almost-linear least-squares (N=200, all free, start=0.5)
# G(I) I=1..N-1: L2, linear part = sum_j X(j) + X(I), constant = N+1
#   contribution = (X(I) + sum_j X(j) - (N+1))^2
# G(N): L2, element PROD = X(1)*...*X(10), constant = 1
#   contribution = (X(1)*X(2)*...*X(10) - 1)^2
function brownal_model(N = 200; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0.5))
    ExaModels.@add_obj(c, (x[i] + sum(x[j] for j in 1:N) - T(N + 1))^2 for i in 1:N-1)
    ExaModels.@add_obj(c, (x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8]*x[9]*x[10] - T(1))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
