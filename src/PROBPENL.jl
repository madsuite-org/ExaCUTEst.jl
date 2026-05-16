# PROBPENL: Probabilistic Penalty problem
# N vars, bounds -5 <= xi <= 5, start 0.5
# EXP2 element: F = (V1+V2)*0.0001/(NG+1)*exp(-V1*V2)
# G(I) for I=1..N-1: EXP2(X(I), X(I+1)) with no group type (identity)
# G(N): L2 group type with linear value sum_i X(I) - 1, scale 0.01
#   → (sum_i X(I) - 1)^2 / 0.01 = 100*(sum_i X(I) - 1)^2... wait
#   Actually: SCALE 0.01 means: contribution = v^2 / scale = v^2 / 0.01
#   But wait, G(N) is N type (objective) and the SIF says:
#   XN G(N) X(I) 1.0 (repeated for I=1..N) with SCALE 0.01
#   XN G(N) 'SCALE' 0.01
#   CONSTANTS: G(N) = 1.0
#   Group type L2: phi(v) = v^2, so contribution = v^2/sigma = v^2/0.01
#   v = sum_i X(I) - 1.0
#   So G(N) contribution = (sum_i X(I) - 1.0)^2 / 0.01
# G(I) for I=1..N-1 (no group type, just element):
#   v = EXP2 element = (V1+V2)*FACT*exp(-V1*V2) where FACT=0.0001/(NG+1)
#   = (X(I)+X(I+1)) * 0.0001/(N-1+1) * exp(-X(I)*X(I+1))
#   = (X(I)+X(I+1)) * 0.0001/N * exp(-X(I)*X(I+1))
#   Note NG = N-1 (RN-1 from SIF)
function probpenl_model(N = 500; T = Float64, backend = nothing, kwargs...)
    NG = T(N - 1)
    FACT = T(0.0001) / (NG + T(1))  # = 0.0001/N

    lvar = fill(T(-5.0), N)
    uvar = fill(T(5.0), N)
    start = fill(T(0.5), N)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # G(I) for I=1..N-1: EXP2 element (identity group type, no scale)
    ExaModels.@add_obj(c, (x[i] + x[i+1]) * FACT * exp(-x[i] * x[i+1]) for i in 1:N-1)

    # G(N): L2 group type with scale 0.01, linear value = sum_i X(I) - 1
    # contribution = (sum_i X(I) - 1)^2 / 0.01
    ExaModels.@add_obj(c, (sum(x[i] for i in 1:N) - T(1.0))^2 / T(0.01))

    return ExaModels.ExaModel(c; kwargs...)
end
