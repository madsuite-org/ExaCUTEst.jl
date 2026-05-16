# LIARWHDNE: Nonlinear equation variant of LIARWHD (N=5000, all free, start=4.0)
# N variables, 2*N equality constraints
# A(I): (X(I)^2 - X(1)) / sqrt(0.25) = 2*(X(I)^2 - X(1)) = 0  (SCALE=sqrt(0.25))
# B(I): X(I) - 1 = 0
# Source: G. Li, SIAM J. Optim., 1990; SIF: Ph. Toint, Aug 1990
function liarwhdne_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(4))
    # A(I) constraints: scale = sqrt(0.25) = 0.5; residual = (x[i]^2 - x[1]) / 0.5
    ExaModels.@add_con(c, T(2) * (x[i]^2 - x[1]) for i in 1:N)
    # B(I) constraints: x[i] - 1 = 0
    ExaModels.@add_con(c, x[i] - T(1) for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
