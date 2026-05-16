# QINGNE: Nonlinear equations version of QING (N=100, all free, start=1.0)
# N variables, N equality constraints
# F(I): X(I)^2 - I = 0  for I=1..N
# Source: SCIPY global optimization benchmark (Qing function)
# SIF: Nick Gould, Jan 2020; classification NOR2-MN-V-V
function qingne_model(N = 100; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1))
    # F(I): x[i]^2 - i = 0
    ExaModels.@add_par(c, RI, T[T(i) for i in 1:N])
    ExaModels.@add_con(c, x[i]^2 - RI[i] for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
