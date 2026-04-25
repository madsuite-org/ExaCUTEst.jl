# SCOSINE: Scaled cosine problem (N=5000 variables, unconstrained)
# NB: scaled version of COSINE
# Source: N. Gould, Nov 1997
# SIF: classification OUR2-AN-V-0
#
# SCALE(I) = exp((I-1)/(N-1) * 12)  for I=1..N
# Start:  x[I] = 1/SCALE(I)
#
# Group G(I), I=1..N-1:
#   linear: ZN G(I) X(I+1) MULT  where AM MULT SCALE(I+1) -0.5
#     AM means multiply: MULT = -0.5 * SCALE(I+1)
#   element: E(I) = SQ(V=X(I), P=SCALE(I)), F = P^2 * V^2 = SCALE(I)^2 * X(I)^2
#   group type COS: contribution = cos(v_G(I))
#
# Group value: v_G(I) = SCALE(I)^2 * x[I]^2 - 0.5 * SCALE(I+1) * x[I+1]
# Objective   = sum_{i=1}^{N-1} cos( SCALE(i)^2 * x[i]^2 - 0.5 * SCALE(i+1) * x[i+1] )
function scosine_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    scal = T(12)
    # SCALE(i) = exp((i-1)/(N-1) * 12)  (N-1 in denominator)
    SCALE = T[exp((i - 1) / T(N - 1) * scal) for i in 1:N]
    # Precompute per-group parameters
    SQ   = T[SCALE[i]^2              for i in 1:N-1]   # element coeff = SCALE(i)^2
    MHN  = T[T(-0.5) * SCALE[i+1]   for i in 1:N-1]   # linear coeff = -0.5 * SCALE(i+1)
    start = T[T(1) / SCALE[i]        for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    ExaModels.@add_par(c, SQ, SQ)
    ExaModels.@add_par(c, MHN, MHN)
    ExaModels.@add_obj(c, cos(SQ[i] * x[i]^2 + MHN[i] * x[i+1]) for i in 1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
