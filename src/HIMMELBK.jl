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
    # For a given CA(I):
    #   sum_{J=1}^{12} 40*B[I]/B[J] * x[I+12] * x[J]
    # + sum_{J=13}^{24} (-C[I]*B[I+12]/B[J]) * x[I] * x[J]
    #
    # Build coefficient arrays indexed by constraint slot k=1..12 (CUTEst order)
    # slot k → SIF index i_sif via: ord = [3,4,5,6,7,8,9,10,11,12,2,1]
    ord = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 1]
    # cf1[k, j] = 40 * B[ord[k]] / B[j]        for j=1..12  (12 terms)
    # cf2[k, j] = -C[ord[k]] * B[ord[k]+12] / B[j+12]  for j=1..12  (i.e. J=13..24)
    cf1 = T[40 * B[ord[k]] / B[j]                for k in 1:12, j in 1:12]
    cf2 = T[-C[ord[k]] * B[ord[k]+12] / B[j+12]  for k in 1:12, j in 1:12]
    ExaModels.@add_par(c, c1, vec(cf1))  # c1[(k-1)*12+j]
    ExaModels.@add_par(c, c2, vec(cf2))  # c2[(k-1)*12+j]

    # Map slot k → SIF variable index i for x[i] and x[i+12]
    # ord[k] gives the SIF constraint index I; x[I+12] is the first factor in part1
    # and x[I] is the first factor in part2
    ord_arr = T.(ord)
    ExaModels.@add_par(c, ordp, ord_arr)

    # Add 12 nonlinear constraints (one per slot k), vectorized:
    # con_k = sum_{j=1}^{12} c1[(k-1)*12+j] * x[ord[k]+12] * x[j]
    #       + sum_{j=1}^{12} c2[(k-1)*12+j] * x[ord[k]]    * x[j+12]
    # We need x[ord[k]+12] and x[ord[k]] inside the loop - these require integer indexing
    # Since ord is a fixed array, we can unroll using a Julia for-loop:
    for k in 1:12
        i_sif = ord[k]  # SIF constraint index
        cf1_k = T[40 * B[i_sif] / B[j] for j in 1:12]
        cf2_k = T[-C[i_sif] * B[i_sif+12] / B[j+12] for j in 1:12]
        ExaModels.@add_par(c, cf1k, cf1_k)
        ExaModels.@add_par(c, cf2k, cf2_k)
        ExaModels.@add_con(c,
            sum(cf1k[j] * x[i_sif+12] * x[j]    for j in 1:12) +
            sum(cf2k[j] * x[i_sif]    * x[j+12]  for j in 1:12))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
