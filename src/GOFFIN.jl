# GOFFIN: Linear minmax problem in 50 variables + slack U
# Source: M.M. Makela, Ph.D. thesis, Jyvaskyla University, 1990
#
# Variables: X(1)..X(50), U (all free)
# Start: X(I) = I - 25.5, U = 0
# Objective: minimize U = x[51]
# Constraints F(I): -U + 50*X(I) - sum_J X(J) ≤ 0, for I=1..50
#   (linear; L type, rhs=0)
function goffin_model(; T = Float64, backend = nothing, kwargs...)
    N = 50
    start_x = T[T(i) - T(25.5) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N + 1;
        lvar  = fill(T(-Inf), N + 1),
        uvar  = fill(T(Inf), N + 1),
        start = [start_x; T(0)])
    ExaModels.@add_obj(c, x[N + 1])
    ExaModels.@add_con(c,
        -x[N + 1] + T(50) * x[i] - sum(x[j] for j in 1:N) for i in 1:N;
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
