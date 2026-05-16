function tointqor_model(; T = Float64, backend = nothing, kwargs...)
    # TOINTQOR: 50-variable QP, no constraints
    # AD SCALE ALPH(I) 1.0 → SCALE = 1/ALPH(I), contribution = v²/SCALE = v²·ALPH(I)
    # Similarly GB(I): contribution = (v-D(I))²·BETA(I)

    alph = T[1.25, 1.40, 2.40, 1.40, 1.75, 1.20, 2.25, 1.20, 1.00, 1.10,
             1.50, 1.60, 1.25, 1.25, 1.20, 1.20, 1.40, 0.50, 0.50, 1.25,
             1.80, 0.75, 1.25, 1.40, 1.60, 2.00, 1.00, 1.60, 1.25, 2.75,
             1.25, 1.25, 1.25, 3.00, 1.50, 2.00, 1.25, 1.40, 1.80, 1.50,
             2.20, 1.40, 1.50, 1.25, 2.00, 1.50, 1.25, 1.40, 0.60, 1.50]

    beta = T[1.0, 1.5, 1.0, 0.1, 1.5, 2.0, 1.0, 1.5, 3.0, 2.0,
             1.0, 3.0, 0.1, 1.5, 0.15, 2.0, 1.0, 0.1, 3.0, 0.1,
             1.2, 1.0, 0.1, 2.0, 1.2, 3.0, 1.5, 3.0, 2.0, 1.0,
             1.2, 2.0, 1.0]

    d = T[-5.0, -5.0, -5.0, -2.5, -6.0, -6.0, -5.0, -6.0, -10.0, -6.0,
          -5.0, -9.0, -2.0, -7.0, -2.5, -6.0, -5.0, -2.0, -9.0, -2.0,
          -5.0, -5.0, -2.5, -5.0, -6.0, -10.0, -7.0, -10.0, -6.0, -5.0,
          -4.0, -4.0, -4.0]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 50;
        lvar = T(-Inf), uvar = T(Inf), start = T(0))

    ExaModels.@add_par(c, alph_par, alph)
    ExaModels.@add_obj(c, alph_par[i] * x[i]^2 for i in 1:50)

    ExaModels.@add_par(c, beta_par, beta)
    ExaModels.@add_par(c, d_par, d)

    ExaModels.@add_obj(c, beta_par[1]  * (-x[31] + x[1]  - d_par[1])^2)
    ExaModels.@add_obj(c, beta_par[2]  * (-x[1]  + x[2]  + x[3]  - d_par[2])^2)
    ExaModels.@add_obj(c, beta_par[3]  * (-x[2]  + x[4]  + x[5]  - d_par[3])^2)
    ExaModels.@add_obj(c, beta_par[4]  * (-x[4]  + x[6]  + x[7]  - d_par[4])^2)
    ExaModels.@add_obj(c, beta_par[5]  * (-x[6]  + x[8]  + x[9]  - d_par[5])^2)
    ExaModels.@add_obj(c, beta_par[6]  * (-x[8]  + x[10] + x[11] - d_par[6])^2)
    ExaModels.@add_obj(c, beta_par[7]  * (-x[10] + x[12] + x[13] - d_par[7])^2)
    ExaModels.@add_obj(c, beta_par[8]  * (-x[12] + x[14] + x[15] - d_par[8])^2)
    ExaModels.@add_obj(c, beta_par[9]  * (-x[11] - x[13] - x[14] + x[16] + x[17] - d_par[9])^2)
    ExaModels.@add_obj(c, beta_par[10] * (-x[16] + x[18] + x[19] - d_par[10])^2)
    ExaModels.@add_obj(c, beta_par[11] * (-x[9]  - x[18] + x[20] - d_par[11])^2)
    ExaModels.@add_obj(c, beta_par[12] * (-x[5]  - x[20] - x[21] - d_par[12])^2)
    ExaModels.@add_obj(c, beta_par[13] * (-x[19] + x[22] + x[23] + x[24] - d_par[13])^2)
    ExaModels.@add_obj(c, beta_par[14] * (-x[23] + x[25] + x[26] - d_par[14])^2)
    ExaModels.@add_obj(c, beta_par[15] * (-x[7]  - x[25] + x[27] + x[28] - d_par[15])^2)
    ExaModels.@add_obj(c, beta_par[16] * (-x[28] + x[29] + x[30] - d_par[16])^2)
    ExaModels.@add_obj(c, beta_par[17] * (-x[29] + x[31] + x[32] - d_par[17])^2)
    ExaModels.@add_obj(c, beta_par[18] * (-x[32] + x[33] + x[34] - d_par[18])^2)
    ExaModels.@add_obj(c, beta_par[19] * (-x[3]  - x[33] + x[35] - d_par[19])^2)
    ExaModels.@add_obj(c, beta_par[20] * (-x[35] + x[21] + x[36] - d_par[20])^2)
    ExaModels.@add_obj(c, beta_par[21] * (-x[36] + x[37] + x[38] - d_par[21])^2)
    ExaModels.@add_obj(c, beta_par[22] * (-x[30] - x[37] + x[39] - d_par[22])^2)
    ExaModels.@add_obj(c, beta_par[23] * (-x[38] - x[39] + x[40] - d_par[23])^2)
    ExaModels.@add_obj(c, beta_par[24] * (-x[40] + x[41] + x[42] - d_par[24])^2)
    ExaModels.@add_obj(c, beta_par[25] * (-x[41] + x[43] + x[44] + x[50] - d_par[25])^2)
    ExaModels.@add_obj(c, beta_par[26] * (-x[44] + x[45] + x[46] + x[47] - d_par[26])^2)
    ExaModels.@add_obj(c, beta_par[27] * (-x[46] + x[48] - d_par[27])^2)
    ExaModels.@add_obj(c, beta_par[28] * (-x[42] - x[45] - x[48] - x[50] + x[49] - d_par[28])^2)
    ExaModels.@add_obj(c, beta_par[29] * (-x[26] - x[34] - x[43] - d_par[29])^2)
    ExaModels.@add_obj(c, beta_par[30] * (-x[15] - x[17] - x[24] - x[47] - d_par[30])^2)
    ExaModels.@add_obj(c, beta_par[31] * (-x[49] - d_par[31])^2)
    ExaModels.@add_obj(c, beta_par[32] * (-x[22] - d_par[32])^2)
    ExaModels.@add_obj(c, beta_par[33] * (-x[27] - d_par[33])^2)

    return ExaModels.ExaModel(c; kwargs...)
end
