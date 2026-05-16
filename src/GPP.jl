# GPP: Geometric programming problem, N=1000 default
# Variables: x(1..N), free, start=1
# Objective: sum_{I<J} exp(x(J)-x(I))
# Constraints:
#   INEQ1(I): x(I) + x(I+1) >= 0
#   INEQ2(I): exp(x(I)) + exp(x(I+1)) <= 20
function gpp_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    x0 = ones(T, N)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # Objective: sum_{I<J} exp(x(J)-x(I))
    pair_data = [(i = i, j = j) for i in 1:(N - 1) for j in (i + 1):N]
    ExaModels.@add_obj(c,
        exp(x[d.j] - x[d.i])
        for d in pair_data)

    # INEQ1(I): x(I) + x(I+1) >= 0
    ExaModels.@add_con(c,
        x[i] + x[i + 1]
        for i in 1:(N - 1);
        lcon = T(0), ucon = T(Inf))

    # INEQ2(I): exp(x(I)) + exp(x(I+1)) - 20 <= 0
    ExaModels.@add_con(c,
        exp(x[i]) + exp(x[i + 1]) - T(20)
        for i in 1:(N - 1);
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
