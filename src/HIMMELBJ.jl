# HIMMELBJ: Chemical equilibrium problem (Himmelblau problem 6, A.P. Jones)
# 45 variables in 7 sets: NS=(4,13,18,3,3,2,2)
# Variable ordering (K=1..7, J=1..NS(K)):
#   K=1: x[1..4]        (X(1,1)..X(4,1))
#   K=2: x[5..17]       (X(1,2)..X(13,2))
#   K=3: x[18..35]      (X(1,3)..X(18,3))
#   K=4: x[36..38]      (X(1,4)..X(3,4))
#   K=5: x[39..41]      (X(1,5)..X(3,5))
#   K=6: x[42..43]      (X(1,6)..X(2,6))
#   K=7: x[44..45]      (X(1,7)..X(2,7))
#
# Objective (identity group, no GROUP TYPE):
#   Linear terms from OBJ group +
#   For each (J,K): X(J,K)*log(X(J,K)) - X(J,K)*log(sum_i X(i,K))
#   The XLOGXn elements (with IV: XX=X, YY=sum Y_i) contribute XX*log(YY) = X*log(sum)
#   with coefficient -1 in GROUP USES.
#
# Constraints: 14 equality constraints (E type), in SIF definition order C1..C8, C11..C16

# Linear objective coefficients: only nonzero entries listed
const HIMMELBJ_LINOBJ = [
    # K=1 (x[1..4]): X2,1=-7.69, X3,1=-11.52, X4,1=-36.60
    (2, -7.69), (3, -11.52), (4, -36.60),
    # K=2 (x[5..17]): X1,2=-10.94, X8,2=2.5966, X9,2=-39.39, X10,2=-21.35, X11,2=-32.84, X12,2=6.26
    (5, -10.94), (12, 2.5966), (13, -39.39), (14, -21.35), (15, -32.84), (16, 6.26),
    # K=3 (x[18..35]): X1,3=10.45, X3,3=-0.5, X7,3=2.2435, X9,3=-39.39, X10,3=-21.49,
    #                   X11,3=-32.84, X12,3=6.12, X15,3=-1.9028, X16,3=-2.8889, X17,3=-3.3622, X18,3=-7.4854
    (18, 10.45), (20, -0.5), (24, 2.2435), (26, -39.39), (27, -21.49),
    (28, -32.84), (29, 6.12), (32, -1.9028), (33, -2.8889), (34, -3.3622), (35, -7.4854),
    # K=4 (x[36..38]): X1,4=-15.639, X3,4=21.81
    (36, -15.639), (38, 21.81),
    # K=5 (x[39..41]): X1,5=-16.79, X3,5=18.9779
    (39, -16.79), (41, 18.9779),
    # K=6 (x[42..43]): X2,6=11.959
    (43, 11.959),
    # K=7 (x[44..45]): X2,7=12.899
    (45, 12.899),
]

function himmelbj_model(; T = Float64, backend = nothing, kwargs...)
    # lvar: default 1e-12, but X(13,2)=x[17] and X(13,3)=x[30] are fixed at specific values
    lvar = fill(T(1e-12), 45)
    lvar[17] = T(0.0155)
    lvar[30] = T(0.0211275)
    uvar = fill(T(Inf), 45)
    uvar[17] = T(0.0155)
    uvar[30] = T(0.0211275)
    start = fill(T(0.1), 45)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    ExaModels.@add_var(c, x, 45; lvar = lvar, uvar = uvar, start = start)

    # Linear objective terms (indexed as above)
    lincoef = T[v for (_, v) in HIMMELBJ_LINOBJ]
    linidx  = [i for (i, _) in HIMMELBJ_LINOBJ]
    ExaModels.@add_par(c, Lc, lincoef)
    ExaModels.@add_par(c, Li, T.(linidx))  # indices as reals for @add_par
    # Add each linear term: sum Lc[k]*x[linidx[k]]
    ExaModels.@add_obj(c, Lc[k] * x[linidx[k]] for k in 1:length(linidx))

    # Entropy terms: for each (J,K), X(J,K)*log(X(J,K)) - X(J,K)*log(sum_i X(i,K))
    # K=1 (x[1..4], sum S1 = x[1]+x[2]+x[3]+x[4])
    ExaModels.@add_obj(c, x[j] * log(x[j]) - x[j] * log(x[1]+x[2]+x[3]+x[4])
        for j in 1:4)
    # K=2 (x[5..17], sum S2 = x[5]+...+x[17])
    ExaModels.@add_obj(c, x[j] * log(x[j]) - x[j] * log(x[5]+x[6]+x[7]+x[8]+x[9]+x[10]+x[11]+x[12]+x[13]+x[14]+x[15]+x[16]+x[17])
        for j in 5:17)
    # K=3 (x[18..35], sum S3 = x[18]+...+x[35])
    ExaModels.@add_obj(c, x[j] * log(x[j]) - x[j] * log(x[18]+x[19]+x[20]+x[21]+x[22]+x[23]+x[24]+x[25]+x[26]+x[27]+x[28]+x[29]+x[30]+x[31]+x[32]+x[33]+x[34]+x[35])
        for j in 18:35)
    # K=4 (x[36..38], sum S4 = x[36]+x[37]+x[38])
    ExaModels.@add_obj(c, x[j] * log(x[j]) - x[j] * log(x[36]+x[37]+x[38])
        for j in 36:38)
    # K=5 (x[39..41], sum S5 = x[39]+x[40]+x[41])
    ExaModels.@add_obj(c, x[j] * log(x[j]) - x[j] * log(x[39]+x[40]+x[41])
        for j in 39:41)
    # K=6 (x[42..43], sum S6 = x[42]+x[43])
    ExaModels.@add_obj(c, x[j] * log(x[j]) - x[j] * log(x[42]+x[43])
        for j in 42:43)
    # K=7 (x[44..45], sum S7 = x[44]+x[45])
    ExaModels.@add_obj(c, x[j] * log(x[j]) - x[j] * log(x[44]+x[45])
        for j in 44:45)

    # Constraints (all equality, E type):
    # C1: x[1]+x[5]+x[18]+x[32]+2*x[33]+3*x[34]+4*x[35] - 0.652981 = 0
    ExaModels.@add_con(c, x[1]+x[5]+x[18]+x[32]+T(2)*x[33]+T(3)*x[34]+T(4)*x[35] - T(0.652981))

    # C2: x[2]+x[6]+x[14]+x[15]+x[16]+x[19]+x[27]+x[28]+x[29]+x[43]+x[45] - 0.281941 = 0
    ExaModels.@add_con(c, x[2]+x[6]+x[14]+x[15]+x[16]+x[19]+x[27]+x[28]+x[29]+x[43]+x[45] - T(0.281941))

    # C3: x[3]+x[7]+x[20] - 3.705233 = 0
    ExaModels.@add_con(c, x[3]+x[7]+x[20] - T(3.705233))

    # C4: x[4]+x[8]+x[13]+x[15]-x[16]+x[21]+x[26]+x[28]-x[29]+x[36]-x[38]+x[39]-x[41]-x[43]-x[45] - 47.00022 = 0
    ExaModels.@add_con(c, x[4]+x[8]+x[13]+x[15]-x[16]+x[21]+x[26]+x[28]-x[29]+x[36]-x[38]+x[39]-x[41]-x[43]-x[45] - T(47.00022))

    # C5: x[4]+x[9]+x[13]+x[14]+x[15]+x[16]+x[22]+x[26]+x[27]+x[28]+x[29] - 47.02972 = 0
    ExaModels.@add_con(c, x[4]+x[9]+x[13]+x[14]+x[15]+x[16]+x[22]+x[26]+x[27]+x[28]+x[29] - T(47.02972))

    # C6: x[10]+x[23] - 0.08005 = 0
    ExaModels.@add_con(c, x[10]+x[23] - T(0.08005))

    # C7: x[11]+x[24] - 0.08813 = 0
    ExaModels.@add_con(c, x[11]+x[24] - T(0.08813))

    # C8: x[12]+x[25] - 0.04829 = 0
    ExaModels.@add_con(c, x[12]+x[25] - T(0.04829))

    # C11: x[31]+x[32]+x[33]+x[34]+x[35] - 0.0022725 = 0
    ExaModels.@add_con(c, x[31]+x[32]+x[33]+x[34]+x[35] - T(0.0022725))

    # C12: x[8]-x[9]-x[10]+x[11]+x[12]+x[14]-2*x[16]-x[17]-4*x[31]-3*x[32]-2*x[33]-x[34] = 0
    ExaModels.@add_con(c, x[8]-x[9]-x[10]+x[11]+x[12]+x[14]-T(2)*x[16]-x[17]-T(4)*x[31]-T(3)*x[32]-T(2)*x[33]-x[34])

    # C13: -x[32]-2*x[33]-3*x[34]-4*x[35]+x[36]+x[37]+x[38] = 0
    ExaModels.@add_con(c, -x[32]-T(2)*x[33]-T(3)*x[34]-T(4)*x[35]+x[36]+x[37]+x[38])

    # C14: x[39]+x[40]+x[41] = 0  (no constant)
    ExaModels.@add_con(c, x[39]+x[40]+x[41])

    # C15: -4*x[38]+x[42]+x[43] = 0
    ExaModels.@add_con(c, -T(4)*x[38]+x[42]+x[43])

    # C16: -4*x[41]+x[44]+x[45] = 0
    ExaModels.@add_con(c, -T(4)*x[41]+x[44]+x[45])

    return ExaModels.ExaModel(c; kwargs...)
end
