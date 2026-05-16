# TQUARTICNE: Nonlinear equation variant of TQUARTIC (N=5000, all free, start=0.1)
# N variables, N equality constraints
# G1: x[1] - 1 = 0
# G(I) for I=2..N: x[1]^2 - x[I]^2 = 0
# Source: Ph. Toint, Dec 1989; SIF: Nick Gould, Jan 2019
# classification NOR2-AN-V-V
function tquarticne_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0.1))
    # G1: x[1] - 1 = 0
    ExaModels.@add_con(c, x[1] - T(1))
    # G(I) for I=2..N: E(1) - E(I) = x[1]^2 - x[i]^2 = 0
    ExaModels.@add_con(c, x[1]^2 - x[i]^2 for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end
