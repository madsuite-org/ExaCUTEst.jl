# BOXPOWER: Box-shaped singular Hessian problem (N=10000, P=9, start=0.99)
# G(1): L2, linear X(1), contribution = X(1)^2
# G(I) I=2..N-1: L2, linear X(1)+X(I)+X(N), contribution = (X(1)+X(I)+X(N))^2
# G(N): POWER type with P=2*(9+1)=20, linear X(N), contribution = X(N)^20
function boxpower_model(N = 20000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0.99))
    ExaModels.@add_obj(c, x[1]^2)
    ExaModels.@add_obj(c, (x[1] + x[i] + x[N])^2 for i in 2:N-1)
    ExaModels.@add_obj(c, x[N]^20)
    return ExaModels.ExaModel(c; kwargs...)
end
