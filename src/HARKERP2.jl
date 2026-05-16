# HARKERP2: Bound-constrained LCP (Harker-Pang problem), N=1000 variables
# min sum_i 0.5*(x_i)^2 * sigma_i_contrib + cross terms
# Groups:
#   S(i): scale=-1, x_i coeff 1.0; HALFL2 → contribution = 0.5*v^2/sigma = 0.5*x_i^2/(-1) = -0.5*x_i^2
#   Q(0): no scale, x_i coeff=-1 for all i; HALFL2 → 0.5*(sum x_i)^2
#   Q(1): scale=0.5, x_i coeff=1 for i=1..N; HALFL2 → (sum x_i)^2
#   Q(j>=2): scale=0.25, x_i coeff=1 for i=j..N; HALFL2 → 2*(sum_{i=j}^N x_i)^2
# Coefficient for pair (i1,i2) with i1 <= i2:
#   diagonal i: from Q groups = -0.5 + 2*i, from S: -0.5 → net = 2*i - 1 → coeff = (2*i-1)/2 for x_i^2
#   off-diagonal i1<i2: from Q groups = -0.5 + 2*i1 → coeff = (-0.5+2*i1) for x_{i1}*x_{i2}
# Start: x_i = i (real)
function harkerp2_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    start = T[i for i in 1:N]
    # off-diagonal pairs: (i1, i2) with i1 < i2, coefficient = -0.5 + 2*i1
    offdiag = [(i1 = i1, i2 = i2, cc = T(-0.5 + 2 * i1)) for i1 in 1:N for i2 in (i1+1):N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; start = start)

    # Diagonal: coeff = (2*i - 1) / 2 = i - 0.5
    ExaModels.@add_obj(c, (T(i) - T(0.5)) * x[i]^2 for i in 1:N)

    # Off-diagonal pairs
    ExaModels.@add_obj(c, d.cc * x[d.i1] * x[d.i2] for d in offdiag)

    return ExaModels.ExaModel(c; kwargs...)
end
