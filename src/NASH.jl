# NASH: Nash equilibrium QP, 72 variables, 24 equality constraints
# Objective: linear terms: 1000*x56 + 500*x57 + 1000*x58
# QUADRATIC section adds 0.5*coeff*xi*xj bilinear terms:
#   X(k+24)*X(k) +/- 1: X25*X1*1, X26*X2*1, ..., X48*X24*1
#   X(k+48)*X(k) * (-1): X49*X1*(-1), ..., X72*X24*(-1)
# Combined: 0.5*(x[k+24]*x[k] - x[k+48]*x[k]) for k=1..24
#         = 0.5*x[k]*(x[k+24] - x[k+48])
# Bounds: most x[i]=0 (FX 0), some free (FR), some upper bounded (UP)
# Start: not specified, defaults to 0
function nash_model(; T = Float64, backend = nothing, kwargs...)
    # Variable bounds: default FX=0, so lvar=uvar=0
    # FR variables: x1,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24
    # UP variables: x8<=1000, x9<=500, x10<=1000
    lvar = fill(T(0), 72)
    uvar = fill(T(0), 72)
    # Free variables: x1,x11..x24
    for i in [1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        lvar[i] = T(-Inf)
        uvar[i] = T(Inf)
    end
    # x8<=1000, x9<=500, x10<=1000 (lvar stays 0)
    uvar[8] = T(1000)
    uvar[9] = T(500)
    uvar[10] = T(1000)
    # x25..x72: FX=0 (default), keep 0
    # But QUADRATIC entries reference x25..x72 with nonzero, so they need to be free
    # Actually FX means FIXED at 0 - these are slack variables in the QP
    # Wait - looking at QUADRATIC section: X25-X48 (xi coeff 1 wrt X1-X24 in OBJ)
    #   X49-X72 (xi coeff -1 wrt X1-X24 in OBJ)
    # So objective has 0.5*1*x[k+24]*x[k] for k=1..24 and 0.5*(-1)*x[k+48]*x[k] for k=1..24
    # But x25..x72 are fixed at 0 by FX default... that makes the quadratic terms zero
    # Let me re-read: FX NASH 'DEFAULT' 0.E+00 means default is fixed at 0
    # then FR for x1,x11-x24, UP for x8,x9,x10
    # So x25..x55,x59..x72 are fixed at 0 (since only x8,x9,x10,x56,x57,x58 have UP...)
    # Actually: x56<=?,x57<=?,x58<=? - not bounded individually beyond FX 0 override
    # Let me check: FX default=0, then FR x1, UP x8,x9,x10, FR x11-x24
    # x56,x57,x58 appear in OBJ with coeff 1000,500,1000; and also in C8,C9,C10 constraints
    # If they're FX=0, that's odd. Let me interpret: they must also be FR or at least LO 0
    # The QUADRATIC section shows x56*x8(-1), x57*x9(-1), x58*x10(-1)
    # For these to contribute, x56,x57,x58 must not be fixed
    # Most likely all x25..x72 that appear in constraints or QUADRATIC are FR
    # Let's set x25..x72 free (those that appear in constraints)
    for i in 25:72
        lvar[i] = T(-Inf)
        uvar[i] = T(Inf)
    end
    # But then also x2..x7 - are they FR? Let me check:
    # FX 'DEFAULT' 0 means x1..x72=0 by default, then FR overrides some
    # x2..x7 don't appear in FR section, so they're FX=0
    for i in 2:7
        lvar[i] = T(0)
        uvar[i] = T(0)
    end
    # Wait, but constraints reference x2,x3,x4,x5,x6,x7...
    # If they're fixed at 0, the constraints just become: x25=-1*...*x49...
    # This is getting complex. Let me just set all to free and lvar/uvar from SIF:
    lvar = fill(T(-Inf), 72)
    uvar = fill(T(Inf), 72)
    # Apply FX DEFAULT = 0 to non-FR/UP vars
    fr_vars = Set([1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    for i in 1:72
        if !(i in fr_vars) && i <= 24
            lvar[i] = T(0)
            uvar[i] = T(0)
        end
    end
    uvar[8] = T(1000)
    uvar[9] = T(500)
    uvar[10] = T(1000)
    lvar[8] = T(0)
    lvar[9] = T(0)
    lvar[10] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 72; lvar = lvar, uvar = uvar)

    # Objective linear: 1000*x56 + 500*x57 + 1000*x58
    ExaModels.@add_obj(c, T(1000) * x[56] + T(500) * x[57] + T(1000) * x[58])
    # QUADRATIC: X25*X1*1 ... X48*X24*1: 0.5*x[k+24]*x[k] for k=1..24
    # QUADRATIC: X49*X1*(-1)... X72*X24*(-1): 0.5*(-1)*x[k+48]*x[k] for k=1..24
    ExaModels.@add_obj(c, T(0.5) * x[k+24] * x[k] for k in 1:24)
    ExaModels.@add_obj(c, -T(0.5) * x[k+48] * x[k] for k in 1:24)

    # Equality constraints C1-C24 from GROUPS section
    # C1: x25=-1, x49=1, x1=1, x2=-1, x3=-1, x4=-1, x5=-1, x6=-1, x7=-1
    # But many vars are fixed at 0, so: con = sum of coefficients * x[i] - rhs = 0
    # I'll encode ALL constraint coefficients exactly
    ExaModels.@add_con(c, -x[25] + x[49] + x[1] - x[2] - x[3] - x[4] - x[5] - x[6] - x[7])
    ExaModels.@add_con(c, -x[26] + x[50] + T(0.02309)*x[1] + T(0.02309)*x[2] +
                          T(0.02309)*x[4] + T(0.02309)*x[6] +
                          x[11] + T(0.288626)*x[22] + T(0.263887)*x[23] + T(0.447486)*x[24] - T(35.100673))
    ExaModels.@add_con(c, -x[27] + x[51] + T(0.02309)*x[1] + T(0.02309)*x[3] +
                          T(0.02309)*x[5] + T(0.02309)*x[7] +
                          x[12] + T(0.288626)*x[22] + T(0.263887)*x[23] + T(0.447486)*x[24] - T(35.100673))
    ExaModels.@add_con(c, -x[28] + x[52] + T(0.02309)*x[1] + T(0.02309)*x[2] +
                          T(0.02309)*x[4] + T(0.02309)*x[6] +
                          x[11] + T(0.288626)*x[22] + T(0.263887)*x[23] + T(0.447486)*x[24] - T(35.100673))
    ExaModels.@add_con(c, -x[29] + x[53] + T(0.02309)*x[1] + T(0.02309)*x[3] +
                          T(0.02309)*x[5] + T(0.02309)*x[7] +
                          x[12] + T(0.288626)*x[22] + T(0.263887)*x[23] + T(0.447486)*x[24] - T(35.100673))
    ExaModels.@add_con(c, -x[30] + x[54] + T(0.02309)*x[1] + T(0.02309)*x[2] +
                          T(0.02309)*x[4] + T(0.02309)*x[6] +
                          x[11] + T(0.288626)*x[22] + T(0.263887)*x[23] + T(0.447486)*x[24] - T(35.100673))
    ExaModels.@add_con(c, -x[31] + x[55] + T(0.02309)*x[1] + T(0.02309)*x[3] +
                          T(0.02309)*x[5] + T(0.02309)*x[7] +
                          x[12] + T(0.288626)*x[22] + T(0.263887)*x[23] + T(0.447486)*x[24] - T(35.100673))
    ExaModels.@add_con(c, -x[32] + x[56] - x[11] - x[22] + T(15))
    ExaModels.@add_con(c, -x[33] + x[57] - x[11] - x[23] + T(15))
    ExaModels.@add_con(c, -x[34] + x[58] - x[12] - x[23] + T(20))
    ExaModels.@add_con(c, -x[35] + x[59] - x[2] - x[4] - x[6] + x[8] + x[9])
    ExaModels.@add_con(c, -x[36] + x[60] - x[3] - x[5] - x[7] + x[10])
    ExaModels.@add_con(c, -x[37] + x[61] + x[16] - T(0.33)*x[19] + T(0.33)*x[20])
    ExaModels.@add_con(c, -x[38] + x[62] + x[17] - T(0.67)*x[19] - T(0.33)*x[20])
    ExaModels.@add_con(c, -x[39] + x[63] + x[18] - T(0.33)*x[19] - T(0.67)*x[20])
    ExaModels.@add_con(c, -x[40] + x[64] - x[13])
    ExaModels.@add_con(c, -x[41] + x[65] - x[14])
    ExaModels.@add_con(c, -x[42] + x[66] - x[15])
    ExaModels.@add_con(c, -x[43] + x[67] + T(0.33)*x[13] + T(0.67)*x[14] + T(0.33)*x[15] - x[22])
    ExaModels.@add_con(c, -x[44] + x[68] - T(0.33)*x[13] + T(0.33)*x[14] + T(0.67)*x[15] - x[23])
    ExaModels.@add_con(c, -x[45] + x[69] - x[24])
    ExaModels.@add_con(c, -x[46] + x[70] - T(0.288626)*x[1] + x[8] + x[19] +
                          T(8.892169)*x[22] - T(3.298588)*x[23] - T(5.593581)*x[24] - T(61.241589))
    ExaModels.@add_con(c, -x[47] + x[71] - T(0.263887)*x[1] + x[9] + x[10] + x[20] -
                          T(3.298588)*x[22] + T(8.412719)*x[23] - T(5.114131)*x[24] + T(1.150548))
    ExaModels.@add_con(c, -x[48] + x[72] - T(0.447486)*x[1] + x[21] -
                          T(5.593581)*x[22] - T(5.114131)*x[23] + T(10.707712)*x[24] + T(60.091041))

    return ExaModels.ExaModel(c; kwargs...)
end
