# DIXON3DQ: Dixon's tridiagonal quadratic (N=10000, all free, start=-1.0)
# G(1): L2, linear X(1)-1 → (X(1)-1)^2
# G(I) I=2..N-1: L2, linear X(I)-X(I+1) → (X(I)-X(I+1))^2
# G(N): L2, linear X(N)-1 → (X(N)-1)^2
function dixon3dq_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(-1))
    ExaModels.@add_obj(c, (x[1] - T(1))^2)
    ExaModels.@add_obj(c, (x[i] - x[i+1])^2 for i in 2:N-1)
    ExaModels.@add_obj(c, (x[N] - T(1))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
