# HATFLDC: Hatfield polytechnic test problem (N=25 variables, unconstrained)
# minimize sum of L2 groups:
#   G(1):  v = X(1) - 1.0,               contribution = (X(1) - 1.0)^2
#   G(I) for I=2..N-1: v = X(I+1) - X(I)^2, contribution = (X(I+1) - X(I)^2)^2
#   G(N):  v = X(N) - 1.0,               contribution = (X(N) - 1.0)^2
# BOUNDS: XU DEFAULT 10.0 (uvar=10.0), XR X(N) (X(N) uvar=Inf); lvar=0 (SIF default)
# START: DEFAULT 0.9
# Source: OPTIMA user manual, Hatfield Polytechnic (1989)
function hatfldc_model(; T = Float64, backend = nothing, kwargs...)
    N = 25
    lvar = fill(T(0.0), N)
    uvar = fill(T(10.0), N)
    lvar[N] = T(-Inf)  # XR X(N): free lower bound on last variable
    uvar[N] = T(Inf)   # XR X(N): no upper bound on last variable

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = T(0.9))

    # G(1): v = X(1) - 1.0
    ExaModels.@add_obj(c, (x[1] - T(1.0))^2)

    # G(I) for I=2..N-1: v = X(I+1) - X(I)^2
    ExaModels.@add_obj(c, (x[i+1] - x[i]^2)^2 for i in 2:N-1)

    # G(N): v = X(N) - 1.0
    ExaModels.@add_obj(c, (x[N] - T(1.0))^2)

    return ExaModels.ExaModel(c; kwargs...)
end
