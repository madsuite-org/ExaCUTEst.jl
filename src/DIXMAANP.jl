# DIXMAANP: Dixon-Maany test problem (version P)
# M=1000 (default), N=3*M=3000 variables, all free, start=2.0
# ALPHA=1.0, BETA=0.26, GAMMA=0.26, DELTA=0.26, K1=2, K2=1, K3=1, K4=2
# GA: AI=(I/N)^2;  GB: BI=0.26*(I/N) (K2=1, varying);
# GC: CI=0.26*(I/N) (K3=1, varying over 1..2M);  GD: DI=0.26*(I/N)^2
function dixmaanp_model(M = 1000; T = Float64, backend = nothing, kwargs...)
    N = 3 * M
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(2))
    ExaModels.@add_par(c, AN, T[(T(i) / T(N))^2 for i in 1:N])
    ExaModels.@add_obj(c, AN[i] * x[i]^2 for i in 1:N)
    ExaModels.@add_obj(c, T(1) for i in 1:1)
    ExaModels.@add_par(c, BN, T[T(0.26) * T(i) / T(N) for i in 1:N-1])
    ExaModels.@add_obj(c, BN[i] * x[i]^2 * (x[i+1] + x[i+1]^2)^2 for i in 1:N-1)
    ExaModels.@add_par(c, CN, T[T(0.26) * T(i) / T(N) for i in 1:2M])
    ExaModels.@add_obj(c, CN[i] * x[i]^2 * x[i+M]^4 for i in 1:2M)
    ExaModels.@add_par(c, DN, T[T(0.26) * (T(i) / T(N))^2 for i in 1:M])
    ExaModels.@add_obj(c, DN[i] * x[i] * x[i+2M] for i in 1:M)
    return ExaModels.ExaModel(c; kwargs...)
end
