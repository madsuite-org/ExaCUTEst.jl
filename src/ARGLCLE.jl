# ARGLCLE: Variable-dimension rank-one linear equations with zero rows/columns
# N=200 variables (default), M=400 equations (default)
# G(1) = 0 (no variables, trivially 0 = 1 → rhs=1 but no variables means infeasible-ish)
# G(I) for I=2..M-1: sum_{j=2}^{N-1} (I-1)*j * x[j] = 1
# G(M) = 0 (objective-type N group, zero contribution)
# SIF: G(1) is XE with no variables; G(M) is XN (objective, no constraints)
# All free variables, start=1.0, rhs=1.0
# Source: More', Garbow, Hillstrom (1981), problem 34
function arglcle_model(N = 200, M = 400; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))

    # G(M) is XN with CONSTANT 1.0 → objective = 0 - 1 = -1
    ExaModels.@add_par(c, _cobj, [-T(1)])
    ExaModels.@add_obj(c, _cobj[1])

    # G(1): XE with no variables → 0 = 1 constraint (always infeasible, but CUTEst includes it)
    ExaModels.@add_con(c, T(0) * x[1] - T(1))

    # G(I) for I=2..M-1: (I-1) * sum_{j=2}^{N-1} j * x[j] = 1
    data = [(ri = T(i - 1),) for i in 2:M-1]
    ExaModels.@add_con(c,
        d.ri * sum(T(j) * x[j] for j in 2:N-1) - T(1)
        for d in data)

    return ExaModels.ExaModel(c; kwargs...)
end
