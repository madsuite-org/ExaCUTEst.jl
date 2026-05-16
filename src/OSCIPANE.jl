# OSCIPANE: Nesterov oscillating path problem (equality constraint version)
# N variables X(1..N), all free
# Groups:
#   Q1 (E): 0.5*x[1] = 0.5  → constraint: 0.5*x[1] - 0.5
#   Q(I) for I=2..N (E): x[I]/RHO - CHEB(X(I-1)) = 0, scale=1/RHO
#     CHEB element: F = 2*TAU^2 - 1
#     Group value: x[I] - CHEB(X(I-1)) * 1.0 (coeff -1.0)
#     With scale 1/RHO: constraint = (x[I] - (2*x[I-1]^2 - 1)) / (1/RHO) * (1/RHO)
#     Actually: group value v = x[I] (linear) + (-1.0)*CHEB(x[I-1])
#            = x[I] - (2*x[I-1]^2 - 1)
#     SCALE 1/RHO: constraint = v * RHO (identity type with scale)
#     Per CLAUDE.md: identity constraint with scale σ → (expr-rhs)/σ
#     So constraint = (x[I] - (2*x[I-1]^2 - 1)) / (1/RHO)
#                   = RHO * (x[I] - 2*x[I-1]^2 + 1)
function oscipane_model(N = 10; T = Float64, backend = nothing, kwargs...)
    RHO = T(500.0)

    start = fill(T(0.0), N)
    start[1] = T(-1.0)
    for i in 2:N
        start[i] = T(1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Q1 (E): 0.5*x[1] - 0.5 = 0
    ExaModels.@add_con(c, T(0.5) * x[1] - T(0.5))

    # Q(I) for I=2..N (E): scale=1/RHO → (x[I] - (2*x[I-1]^2 - 1)) / (1/RHO)
    #                     = RHO * (x[I] - 2*x[I-1]^2 + 1)
    ExaModels.@add_con(c, RHO * (x[i] - T(2) * x[i-1]^2 + T(1)) for i in 2:N)

    return ExaModels.ExaModel(c; kwargs...)
end
