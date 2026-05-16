# PRODPL1: Production planning QP, 60 variables, 29 constraints
# Same structure as PRODPL0 but with hardcoded element uses
# SMOOTH elements: (U1-U2)^2 - 0.1*U2^2 <= 0  (rho^2=0.1, not 0.01!)
# Source: L. Escudero, SIF input: A.R. Conn, March 1991
function prodpl1_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend)
    # 60 variables: x[1..20] = X vars, x[21..60] = I/Y inventory vars
    ExaModels.@add_var(c, x, 60; lvar = T(0))
    # Linear objective
    ExaModels.@add_obj(c,
        T(1)*x[21] + T(2)*x[22] + T(2)*x[23] + T(3)*x[24] +
        T(2)*x[25] + T(4)*x[26] + T(3)*x[27] + T(5)*x[28] +
        T(1)*x[29] + T(2)*x[30] + T(2)*x[31] + T(3)*x[32] +
        T(3)*x[33] + T(2)*x[34] + T(4)*x[35] + T(5)*x[36] +
        T(1)*x[37] + T(2)*x[38] + T(2)*x[39] + T(3)*x[40] +
        T(3)*x[41] + T(2)*x[42] + T(4)*x[43] + T(5)*x[44] +
        T(1)*x[45] + T(2)*x[46] + T(2)*x[47] + T(3)*x[48] +
        T(3)*x[49] + T(2)*x[50] + T(4)*x[51] + T(5)*x[52] +
        T(1)*x[53] + T(2)*x[54] + T(2)*x[55] + T(3)*x[56] +
        T(3)*x[57] + T(2)*x[58] + T(4)*x[59] + T(5)*x[60])
    # K constraints (L type)
    ExaModels.@add_con(c, x[1] + x[2] + x[3] + x[4] - T(3); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, x[5] + x[6] + x[7] + x[8] - T(6); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, x[9] + x[10] + x[11] + x[12] - T(10); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, x[13] + x[14] + x[15] + x[16] - T(2000); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, x[17] + x[18] + x[19] + x[20] - T(18); lcon = T(-Inf), ucon = T(0))
    # D equality constraints
    ExaModels.@add_con(c, x[1] - x[21] + x[22] - T(1))
    ExaModels.@add_con(c, x[2] - x[23] + x[24] - T(1))
    ExaModels.@add_con(c, x[3] - x[25] + x[26] - T(1))
    ExaModels.@add_con(c, x[4] - x[27] + x[28] - T(1))
    ExaModels.@add_con(c, x[5] + x[21] - x[29] + x[30] - T(2.667))
    ExaModels.@add_con(c, x[6] + x[23] - x[31] + x[32] - T(1.667))
    ExaModels.@add_con(c, x[7] + x[25] - x[33] + x[34] - T(2.667))
    ExaModels.@add_con(c, x[8] + x[27] - x[35] + x[36] - T(3.333))
    ExaModels.@add_con(c, x[9] + x[29] - x[37] + x[38] - T(2.667))
    ExaModels.@add_con(c, x[10] + x[31] - x[39] + x[40] - T(2.0))
    ExaModels.@add_con(c, x[11] + x[33] - x[41] + x[42] - T(3.0))
    ExaModels.@add_con(c, x[12] + x[35] - x[43] + x[44] - T(3.0))
    ExaModels.@add_con(c, x[13] + x[37] - x[45] + x[46] - T(2.667))
    ExaModels.@add_con(c, x[14] + x[39] - x[47] + x[48] - T(2.667))
    ExaModels.@add_con(c, x[15] + x[41] - x[49] + x[50] - T(2.667))
    ExaModels.@add_con(c, x[16] + x[43] - x[51] + x[52] - T(2.667))
    ExaModels.@add_con(c, x[17] + x[45] - x[53] + x[54] - T(2.667))
    ExaModels.@add_con(c, x[18] + x[47] - x[55] + x[56] - T(2.333))
    ExaModels.@add_con(c, x[19] + x[49] - x[57] + x[58] - T(2.333))
    ExaModels.@add_con(c, x[20] + x[51] - x[59] + x[60] - T(2.333))
    # SMOOTH constraints: PRODPL1 uses F = U1MU2^2 - 0.1*U2^2 (rho_sq=0.1)
    # SMOOTH1: NLE1 with U1=x[5]+x[6]+x[7]+x[8], U2=x[1]+x[2]+x[3]+x[4]
    ExaModels.@add_con(c,
        (x[5]+x[6]+x[7]+x[8] - (x[1]+x[2]+x[3]+x[4]))^2 -
        T(0.1)*(x[1]+x[2]+x[3]+x[4])^2;
        lcon = T(-Inf), ucon = T(0))
    # SMOOTH2: NLE2
    ExaModels.@add_con(c,
        (x[9]+x[10]+x[11]+x[12] - (x[5]+x[6]+x[7]+x[8]))^2 -
        T(0.1)*(x[5]+x[6]+x[7]+x[8])^2;
        lcon = T(-Inf), ucon = T(0))
    # SMOOTH3: NLE3
    ExaModels.@add_con(c,
        (x[13]+x[14]+x[15]+x[16] - (x[9]+x[10]+x[11]+x[12]))^2 -
        T(0.1)*(x[9]+x[10]+x[11]+x[12])^2;
        lcon = T(-Inf), ucon = T(0))
    # SMOOTH4: NLE4
    ExaModels.@add_con(c,
        (x[17]+x[18]+x[19]+x[20] - (x[13]+x[14]+x[15]+x[16]))^2 -
        T(0.1)*(x[13]+x[14]+x[15]+x[16])^2;
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
