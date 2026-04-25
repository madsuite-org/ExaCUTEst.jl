# HIMMELBK: Blending problem for multi-component mixtures (Paviani / Himmelblau problem 20)
# 24 variables X(1)..X(24), all in [0, Inf], start at 0.04
# Linear objective, with 14 linear/nonlinear equality constraints
#
# Constraint ordering (empirically verified):
#   cons[1]  = CA14 (linear)
#   cons[2]  = CA13 (linear)
#   cons[3..14] = CA(3), CA(4), ..., CA(12), CA(2), CA(1)  (nonlinear)
#
# CA(I) = sum_{J=1}^{12} 40*B(I)/B(J) * x[I+12]*x[J]
#        + sum_{J=13}^{24} (-C(I)*B(I+12)/B(J)) * x[I]*x[J]
# CA13   = sum_{k=1}^{24} x[k] - 1.0
# CA14   = sum_{I=1}^{12} x[I]/D(I) + sum_{I=1}^{12} (F/B(I+12)) * x[I+12] - 1.671

function himmelbk_model(; T = Float64, backend = nothing, kwargs...)
    B = T[44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097,
          44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097]
    C = T[123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64]
    D = T[123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64]
    F = T(142.22471)

    obj_coef = T[0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09,
                 0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    ExaModels.@add_var(c, x, 24;
        lvar  = fill(T(0.0), 24),
        uvar  = fill(T(Inf), 24),
        start = fill(T(0.04), 24))

    # Objective: linear sum
    ExaModels.@add_par(c, oc, obj_coef)
    ExaModels.@add_obj(c, oc[k] * x[k] for k in 1:24)

    # CA14 (linear): sum_{I=1}^{12} x[I]/D[I] + sum_{I=1}^{12} (F/B[I+12]) * x[I+12] - 1.671
    d_inv  = T[1 / D[i] for i in 1:12]
    fb_inv = T[F / B[i+12] for i in 1:12]
    ExaModels.@add_par(c, dinv, d_inv)
    ExaModels.@add_par(c, fbinv, fb_inv)
    ExaModels.@add_con(c,
        sum(dinv[i] * x[i]    for i in 1:12) +
        sum(fbinv[i] * x[i+12] for i in 1:12) - T(1.671))

    # CA13 (linear): sum_{k=1}^{24} x[k] - 1.0
    ExaModels.@add_con(c, sum(x[k] for k in 1:24) - T(1.0))

    # Nonlinear constraints CA(I) in CUTEst order: [3,4,5,6,7,8,9,10,11,12,2,1]
    # For CA(I): sum_{J=1}^{12} 40*B[I]/B[J] * x[I+12]*x[J]
    #          + sum_{J=13}^{24} (-C[I]*B[I+12]/B[J]) * x[I]*x[J]
    for i_sif in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 1]
        cf1_k = T[40 * B[i_sif] / B[j]           for j in 1:12]
        cf2_k = T[-C[i_sif] * B[i_sif+12] / B[j+12] for j in 1:12]
        ExaModels.@add_par(c, cf1k, cf1_k)
        ExaModels.@add_par(c, cf2k, cf2_k)
        ExaModels.@add_con(c,
            sum(cf1k[j] * x[i_sif+12] * x[j]    for j in 1:12) +
            sum(cf2k[j] * x[i_sif]    * x[j+12]  for j in 1:12))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
