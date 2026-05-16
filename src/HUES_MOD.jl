# HUES-MOD: Astronomy inverse problem (Hestis 1992), K=5000 variables
# min (1/K) * sum_{i=1}^K M(i)^2
# s.t. sum_i M(i) * C1(i) = 1835.2, sum_i M(i) * C2(i) = 909.8
# C1(i) = (i*h)^3/3 - ((i-1)*h)^3/3, C2(i) = (i*h)^5/5 - ((i-1)*h)^5/5, h=1/K
# lvar=0 (SIF default), start=1.0
function hues_mod_model(K = 5000; T = Float64, backend = nothing, kwargs...)
    h = T(1) / K
    coeff1 = T[(i * h)^3 / 3 - ((i-1) * h)^3 / 3 for i in 1:K]
    coeff2 = T[(i * h)^5 / 5 - ((i-1) * h)^5 / 5 for i in 1:K]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, K; start = T(1))

    # Objective: (1/K) * sum M(i)^2
    # SQ element (F=U1^2) with coeff 1/K; group OBJ(i) has scale=1/K? No:
    # Each group OBJ(i) has element E(i) with coeff 1/K → contribution = (1/K) * x_i^2
    # (no group scale, no group type → identity, contribution = v_i = (1/K)*x_i^2)
    # Actually: element SQ: F=U1^2, so element value = x_i^2
    # Group OBJ(i) is identity type (no GROUP TYPE) → contribution = group_value = (1/K)*x_i^2
    # (element used with coefficient 1/K)
    ExaModels.@add_par(c, INV_K, T[T(1) / K for _ in 1:K])
    ExaModels.@add_obj(c, INV_K[i] * x[i]^2 for i in 1:K)

    # E1: sum_i C1(i)*M(i) = 1835.2
    ExaModels.@add_par(c, C1, coeff1)
    ExaModels.@add_con(c, sum(C1[i] * x[i] for i in 1:K) - T(1835.2))

    # E2: sum_i C2(i)*M(i) = 909.8
    ExaModels.@add_par(c, C2, coeff2)
    ExaModels.@add_con(c, sum(C2[i] * x[i] for i in 1:K) - T(909.8))

    return ExaModels.ExaModel(c; kwargs...)
end
