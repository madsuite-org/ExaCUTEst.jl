# DISC2: Minimum disc radius for polygon covering interior discs
# Source: W. Pulleyblank, private communication, 1991
# SIF input: A.R. Conn, November 1991
# classification LQR2-MY-29-23
#
# 29 variables, 23 constraints
# SIF variable declaration order: EPSILON, then for I=1..11: U(I), V(I), then ALPHA(1..6)
# So variable layout (1-indexed):
#   x[1] = EPSILON
#   x[2I] = U(I), x[2I+1] = V(I) for I=1..11
#   x[24..29] = ALPHA(1..6)
#
# CIRCLE element: F = (X-P1)^2 + (Y-P2)^2 - Z^2
# LINE element: IV U = X - Y, IV V = -Z; F = U*V = (X-Y)*(-Z)
#
# B(I) equality (I=1..5): (U(I)-XI[I])^2 + (V(I)-YI[I])^2 - EPSILON^2 = 0
# Collinearity: B_ijk = linear_part + element_contribution = 0
# B(I) inequality (I=6..11): (U(I)-XI[I])^2 + (V(I)-YI[I])^2 - EPSILON^2 <= 0
#
# CUTEst constraint ordering (empirically determined via sensitivity analysis):
#   g[1..5]:  B(1)..B(5)  (CIRCLE equalities, forward)
#   g[6]:  C5111  (V11-V5) + (V1-V5)*(-ALPHA5)
#   g[7]:  B5111  (U11-U5) + (U1-U5)*(-ALPHA5)
#   g[8]:  C5101  (V10-V5) + (V1-V5)*(-ALPHA5)
#   g[9]:  B5101  (U10-U5) + (U1-U5)*(-ALPHA5)
#   g[10]: C495   (V9-V4) + (V5-V4)*(-ALPHA4)
#   g[11]: B495   (U9-U4) + (U5-U4)*(-ALPHA4)
#   g[12]: B162   (U6-U1) + (U2-U1)*(-ALPHA1)
#   g[13]: C162   (V6-V1) + (V2-V1)*(-ALPHA1)
#   g[14]: B273   (U7-U2) + (U3-U2)*(-ALPHA2)
#   g[15]: C273   (V7-V2) + (V3-V2)*(-ALPHA2)
#   g[16]: B384   (U8-U3) + (U4-U3)*(-ALPHA3)
#   g[17]: C384   (V8-V3) + (V4-V3)*(-ALPHA3)
#   g[18]: B(11) (CIRCLE ineq, reversed order)
#   g[19]: B(10)
#   g[20]: B(9)
#   g[21]: B(8)
#   g[22]: B(7)
#   g[23]: B(6)
function disc2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    # Node coordinates (XI[I], YI[I]) for I=1..11
    ExaModels.@add_par(c, XI, T[0.0, 8.0, 12.0, 8.0, 0.0, 4.0, 8.0, 8.0, 4.0, 2.0, 2.0])
    ExaModels.@add_par(c, YI, T[10.0, 10.0, 5.0, 0.0, 0.0, 8.0, 7.0, 3.0, 1.0, 3.0, 6.0])

    # Variable layout: x[1]=EPSILON, x[2I]=U(I), x[2I+1]=V(I) for I=1..11, x[24..29]=ALPHA(1..6)
    # EPSILON: [0,3], U/V: free, ALPHA: [0,1]
    lvar = fill(T(-Inf), 29)
    uvar = fill(T(Inf), 29)
    x0   = fill(T(5), 29)
    lvar[1] = T(0); uvar[1] = T(3); x0[1] = T(0.5)   # EPSILON
    for j in 1:6
        lvar[23+j] = T(0); uvar[23+j] = T(1); x0[23+j] = T(0)  # ALPHA(j)
    end

    ExaModels.@add_var(c, x, 29; lvar = lvar, uvar = uvar, start = x0)

    # Objective: minimize EPSILON = x[1]
    ExaModels.@add_obj(c, x[1])

    # Helper macros: U(I) = x[2I], V(I) = x[2I+1], EPSILON = x[1]
    # CIRCLE B(I) = (U(I)-XI[I])^2 + (V(I)-YI[I])^2 - EPSILON^2
    # LINE collinearity = linear_part + (X-Y)*(-Z)

    # g[1..5]: B(1..5) CIRCLE equalities (CUTEst ordering: forward i=1..5)
    ExaModels.@add_con(c, (x[2i] - XI[i])^2 + (x[2i+1] - YI[i])^2 - x[1]^2 for i in 1:5)

    # g[6]: C5111 = (V11-V5) + (V1-V5)*(-ALPHA5)
    #   = (x[23]-x[11]) + (x[3]-x[11])*(-x[28])
    ExaModels.@add_con(c, (x[23] - x[11]) + (x[3] - x[11]) * (-x[28]))

    # g[7]: B5111 = (U11-U5) + (U1-U5)*(-ALPHA5)
    #   = (x[22]-x[10]) + (x[2]-x[10])*(-x[28])
    ExaModels.@add_con(c, (x[22] - x[10]) + (x[2] - x[10]) * (-x[28]))

    # g[8]: C5101 = (V10-V5) + (V1-V5)*(-ALPHA5)
    #   = (x[21]-x[11]) + (x[3]-x[11])*(-x[28])
    ExaModels.@add_con(c, (x[21] - x[11]) + (x[3] - x[11]) * (-x[28]))

    # g[9]: B5101 = (U10-U5) + (U1-U5)*(-ALPHA5)
    #   = (x[20]-x[10]) + (x[2]-x[10])*(-x[28])
    ExaModels.@add_con(c, (x[20] - x[10]) + (x[2] - x[10]) * (-x[28]))

    # g[10]: C495 = (V9-V4) + (V5-V4)*(-ALPHA4)
    #   = (x[19]-x[9]) + (x[11]-x[9])*(-x[27])
    ExaModels.@add_con(c, (x[19] - x[9]) + (x[11] - x[9]) * (-x[27]))

    # g[11]: B495 = (U9-U4) + (U5-U4)*(-ALPHA4)
    #   = (x[18]-x[8]) + (x[10]-x[8])*(-x[27])
    ExaModels.@add_con(c, (x[18] - x[8]) + (x[10] - x[8]) * (-x[27]))

    # g[12]: B162 = (U6-U1) + (U2-U1)*(-ALPHA1)
    #   = (x[12]-x[2]) + (x[4]-x[2])*(-x[24])
    ExaModels.@add_con(c, (x[12] - x[2]) + (x[4] - x[2]) * (-x[24]))

    # g[13]: C162 = (V6-V1) + (V2-V1)*(-ALPHA1)
    #   = (x[13]-x[3]) + (x[5]-x[3])*(-x[24])
    ExaModels.@add_con(c, (x[13] - x[3]) + (x[5] - x[3]) * (-x[24]))

    # g[14]: B273 = (U7-U2) + (U3-U2)*(-ALPHA2)
    #   = (x[14]-x[4]) + (x[6]-x[4])*(-x[25])
    ExaModels.@add_con(c, (x[14] - x[4]) + (x[6] - x[4]) * (-x[25]))

    # g[15]: C273 = (V7-V2) + (V3-V2)*(-ALPHA2)
    #   = (x[15]-x[5]) + (x[7]-x[5])*(-x[25])
    ExaModels.@add_con(c, (x[15] - x[5]) + (x[7] - x[5]) * (-x[25]))

    # g[16]: B384 = (U8-U3) + (U4-U3)*(-ALPHA3)
    #   = (x[16]-x[6]) + (x[8]-x[6])*(-x[26])
    ExaModels.@add_con(c, (x[16] - x[6]) + (x[8] - x[6]) * (-x[26]))

    # g[17]: C384 = (V8-V3) + (V4-V3)*(-ALPHA3)
    #   = (x[17]-x[7]) + (x[9]-x[7])*(-x[26])
    ExaModels.@add_con(c, (x[17] - x[7]) + (x[9] - x[7]) * (-x[26]))

    # g[18..23]: B(11), B(10), ..., B(6) CIRCLE inequalities (CUTEst: reversed i=11..6)
    ExaModels.@add_con(c, (x[2i] - XI[i])^2 + (x[2i+1] - YI[i])^2 - x[1]^2 for i in 11:-1:6;
                       lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
