# CHEBYQAD: Chebyquad problem (N variables, unconstrained)
# Source: More', Garbow, Hillstrom (1981), problem 35; Buckley #133
# SIF: N=M=100 (default), groups G(I) for I=1..M with L2 group type (no scale)
#
# Each group G(I):
#   v_I = (1/N) * sum_J cos(I*acos(2*X(J)-1)) - c_I
#   c_I = 0 if I odd, -1/(I^2-1) if I even
#   contribution = v_I^2
#
# Bounds: X(J) in [0,1], start X(J) = J/(N+1)
#
# Uses @add_par for RI and CI so all M groups share one compiled type.
# The O(N^2) Hessian sparsity computation is done once for the inner sum.
# Use N=10 in tests (see runtests.jl override) to keep CI times reasonable.
function chebyqad_model(N = 100; T = Float64, backend = nothing, kwargs...)
    M = N
    start = T[j / (N + 1) for j in 1:N]
    c_data = T[isodd(i) ? T(0) : T(-1 / (i^2 - 1)) for i in 1:M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0), uvar = T(1), start = start)
    ExaModels.@add_par(c, CI, c_data)
    ExaModels.@add_par(c, RI, T[T(i) for i in 1:M])
    ExaModels.@add_obj(
        c,
        (sum(cos(RI[i] * acos(T(2) * x[j] - T(1))) for j in 1:N) / T(N) - CI[i])^2
        for i in 1:M
    )
    return ExaModels.ExaModel(c; kwargs...)
end
