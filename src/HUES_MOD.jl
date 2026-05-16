# HUES-MOD: Astronomy inverse problem (Hestis 1992), K=5000 variables
# min (1/K) * sum_{i=1}^K M(i)^2
# s.t. sum_i M(i) * C1(i) = 1835.2, sum_i M(i) * C2(i) = 909.8
# C1(i) = (i*h)^3/3 - ((i-1)*h)^3/3, C2(i) = (i*h)^5/5 - ((i-1)*h)^5/5, h=1/K
# lvar=0 (SIF default), start=1.0
function hues_mod_model(K = 5000; T = Float64, backend = nothing, kwargs...)
    h = T(1) / K
    coeff1 = T[(i * h)^3 / 3 - ((i-1) * h)^3 / 3 for i in 1:K]
    coeff2 = T[(i * h)^5 / 5 - ((i-1) * h)^5 / 5 for i in 1:K]
    inv_K = T(1) / T(K)

    # Use K+2 variables: x[1:K] = M(i), x[K+1] = slack for con1, x[K+2] = slack for con2
    # Constraints: x[K+1] = 1835.2, x[K+2] = 909.8
    # Augmentation: x[K+1] += C1[i]*x[i], x[K+2] += C2[i]*x[i]  for i=1:K
    # But wait — CUTEst expects nvar=K, not K+2.

    # Alternative: use the existing sum() approach with lcon/ucon handling
    # The compilation warning is just slow, not broken. For K=5000, it will compile eventually.
    # But for practical purposes, let us limit to the sum() approach which is correct:
    
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, K; lvar = T(0), uvar = T(Inf), start = T(1))

    ExaModels.@add_par(c, INV_K, T[inv_K for _ in 1:K])
    ExaModels.@add_obj(c, INV_K[i] * x[i]^2 for i in 1:K)

    ExaModels.@add_par(c, C1, coeff1)
    ExaModels.@add_par(c, C2, coeff2)

    # Use constraint augmentation pattern
    g1 = ExaModels.@add_con(c, C1[1] * x[1] - T(1835.2))
    ExaModels.@add_con!(c, g1[1] += C1[i] * x[i] for i in 2:K)

    g2 = ExaModels.@add_con(c, C2[1] * x[1] - T(909.8))
    ExaModels.@add_con!(c, g2[1] += C2[i] * x[i] for i in 2:K)

    return ExaModels.ExaModel(c; kwargs...)
end
