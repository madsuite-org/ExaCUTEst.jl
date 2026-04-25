# DQDRTIC: diagonal quadratic (N=5000 variables, unconstrained)
# minimize sum_{i=1}^{N-2} [100*x[i+1]^2 + 100*x[i+2]^2 + x[i]^2]
# All variables free, starting at 3.0
# Source: Toint (1983), SIF: Toint, Dec 1989
# SIF: NGS=N-2 groups of L2 type, each group with SCALE=0.01 → coeff = 1/0.01 = 100
#   A(I): SCALE=0.01, linear X(I+1) → contribution = X[i+1]^2 / 0.01 = 100*X[i+1]^2
#   B(I): SCALE=0.01, linear X(I+2) → contribution = X[i+2]^2 / 0.01 = 100*X[i+2]^2
#   C(I): no scale, linear X(I) → contribution = X[i]^2
function dqdrtic_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    NGS = N - 2
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(3))
    ExaModels.@add_obj(c, T(100) * x[i+1]^2 for i in 1:NGS)  # A(I) groups
    ExaModels.@add_obj(c, T(100) * x[i+2]^2 for i in 1:NGS)  # B(I) groups
    ExaModels.@add_obj(c, x[i]^2 for i in 1:NGS)             # C(I) groups
    return ExaModels.ExaModel(c; kwargs...)
end
