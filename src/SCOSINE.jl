# SCOSINE: Scaled cosine problem (N=5000 variables, unconstrained)
# NB: scaled version of COSINE
# Source: N. Gould, Nov 1997
# SIF: classification OUR2-AN-V-0
#
# SCALE(I) = exp((I-1)/(N-1) * 12)  for I=1..N
# Start:  x[I] = 1 / SCALE(I)
#
# Element SQ(V=X(I), P=SCALE(I)):  F = P^2 * V^2 = SCALE(I)^2 * X(I)^2
# Group G(I) linear coeff on X(I+1):  MULT = SCALE(I+1)^{-0.5} = 1/sqrt(SCALE(I+1))
# Group value:  v_G(I) = SCALE(I)^2 * X(I)^2 + X(I+1)/sqrt(SCALE(I+1))
# Group type COS: contribution = cos(v_G(I))
#
# Objective = sum_{i=1}^{N-1} cos( SCALE(i)^2 * x[i]^2 + x[i+1] / sqrt(SCALE(i+1)) )
function scosine_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    scal = T(12)
    # SCALE(i) = exp((i-1)/(N-1) * 12) — note N-1 in denominator (RN-1 = float(N-1))
    SCALE = T[exp((i - 1) / T(N - 1) * scal) for i in 1:N]
    # Precompute per-group parameters: sq_scale = SCALE(i)^2, inv_sqrt_next = 1/sqrt(SCALE(i+1))
    SQ    = T[SCALE[i]^2               for i in 1:N-1]
    ISQN  = T[T(1) / sqrt(SCALE[i+1]) for i in 1:N-1]
    start = T[T(1) / SCALE[i]          for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    ExaModels.@add_par(c, SQ, SQ)
    ExaModels.@add_par(c, ISQN, ISQN)
    ExaModels.@add_obj(c, cos(SQ[i] * x[i]^2 + ISQN[i] * x[i+1]) for i in 1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
