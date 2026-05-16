# SMBANK: Small Bank Balancing problem (Thai model), also named "MB64"
# 117 arcs (variables), 64 nodes (equality constraints)
# Objective: sum_k X_k * (log(X_k / C1_k) - 1)  for arcs 1-84 and 117
# Constraints: network flow conservation at each node (each node's net flow = 0)
# Bounds: DEFAULT lvar=0.1, uvar=999999; X85-X116 lvar=-99999
function smbank_model(; T = Float64, backend = nothing, kwargs...)
    NARCS  = 117

    # C1 values for arcs with objective contribution (E1-E84, E117)
    C1_data = T[
        1.75726e5, 1.40841e5, 1.28910e4, 2.73191e5, 2.73191e5,
        1.28910e4, 1.40841e5, 1.75726e5, 4.75900e3, 3.80000e2,
        3.85900e3, 5.20758e5, 9.94000e3, 6.16400e3, 6.31180e4,
        2.73800e3, 7.10830e4, 1.01530e4, 9.29400e3, 4.36700e3,
        4.42511e5, 8.46680e4, 4.85240e4, 2.64500e3, 6.52100e4,
        7.15280e4, 7.43460e4, 2.26846e5, 5.87250e4, 4.10017e5,
        5.27740e4, 3.18470e4, 4.16672e5, 2.28906e5, 4.37902e5,
        5.35200e4, 4.34014e5, 1.19280e4, 4.67470e4, 2.22840e4,
        1.90930e4, 1.31949e5, 2.99470e4, 9.82000e2, 1.75800e5,
        3.02740e4, 1.61094e5, 7.20300e3, 1.73359e5, 1.89710e4,
        3.29050e4, 2.19780e4, 9.61300e3, 7.72000e3, 2.84600e3,
        1.95220e4, 1.14482e5, 5.99600e3, 8.33760e4, 6.32950e4,
        7.46190e4, 1.17681e5, 3.09500e3, 1.40757e5, 6.00350e4,
        2.54350e4, 7.75870e4, 5.88840e4, 3.18470e4, 1.59000e2,
        3.24100e3, 2.13800e3, 1.65830e4, 9.29000e2, 1.73420e4,
        7.46000e2, 2.78850e4, 2.06000e3, 2.54350e4, 5.78970e4,
        1.24174e5, 2.16600e3, 2.00000e0, 2.59000e2,
        1.44060e4  # arc X117 (index 85 in this array)
    ]

    # Bounds
    lvar = fill(T(0.1), NARCS)
    uvar = fill(T(999999), NARCS)
    for k in 85:116; lvar[k] = T(-99999); end

    # Start point (XV SMBANK 'DEFAULT' 0.1, then individual overrides)
    x0 = fill(T(0.1), NARCS)
    x0[4]=T(0.7); x0[5]=T(0.7); x0[12]=T(0.7); x0[17]=T(0.9)
    x0[21]=T(0.6); x0[22]=T(0.9); x0[28]=T(0.4); x0[30]=T(0.4)
    x0[31]=T(0.4); x0[33]=T(1.0); x0[34]=T(0.5); x0[35]=T(0.5)
    x0[36]=T(0.5); x0[37]=T(1.1); x0[42]=T(0.3); x0[47]=T(0.2)
    x0[61]=T(0.7); x0[63]=T(0.2); x0[64]=T(0.3); x0[65]=T(0.2)
    x0[67]=T(0.2); x0[68]=T(0.2); x0[81]=T(0.2)
    x0[88]=T(0.7); x0[89]=T(0.9); x0[90]=T(1.0); x0[91]=T(0.3)
    x0[92]=T(1.3); x0[93]=T(0.6); x0[94]=T(0.9); x0[95]=T(0.4)
    x0[96]=T(0.5); x0[97]=T(0.5); x0[98]=T(0.4); x0[99]=T(1.1)
    x0[100]=T(0.5); x0[101]=T(0.5); x0[102]=T(0.5); x0[103]=T(1.1)
    x0[104]=T(0.7); x0[105]=T(0.8); x0[106]=T(0.7); x0[107]=T(1.2)
    x0[108]=T(0.2); x0[109]=T(0.3); x0[110]=T(0.2); x0[112]=T(0.2)
    x0[113]=T(0.2); x0[115]=T(0.9); x0[116]=T(0.7)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NARCS; lvar = lvar, uvar = uvar, start = x0)

    # Objective: x[k]*(log(x[k]/c1[k])-1) for arcs 1..84 (c1 indexed 1..84)
    ExaModels.@add_par(c, c1, C1_data)
    ExaModels.@add_obj(c, x[i] * (log(x[i] / c1[i]) - T(1)) for i in 1:84)
    # arc 117 uses c1[85] = 1.44060e4
    ExaModels.@add_obj(c, x[117] * (log(x[117] / c1[85]) - T(1)))

    # Network flow constraints: one equality per node
    # sign convention: incoming arc to node = +1, outgoing arc = -1
    # Node 1: x[85] - x[1] = 0
    ExaModels.@add_con(c, x[85] - x[1])
    # Node 2: x[86] - x[2] = 0
    ExaModels.@add_con(c, x[86] - x[2])
    # Node 3: x[87] - x[3] = 0
    ExaModels.@add_con(c, x[87] - x[3])
    # Node 4: x[88] - x[4] = 0
    ExaModels.@add_con(c, x[88] - x[4])
    # Node 5: x[89]+x[117] - (x[5]+x[6]+x[7]+x[8]) = 0
    ExaModels.@add_con(c, x[89]+x[117] - x[5]-x[6]-x[7]-x[8])
    # Node 6: x[90] - (x[9]+x[10]+x[11]+x[12]) = 0
    ExaModels.@add_con(c, x[90] - x[9]-x[10]-x[11]-x[12])
    # Node 7: x[91] - (x[13]+x[14]+x[15]) = 0
    ExaModels.@add_con(c, x[91] - x[13]-x[14]-x[15])
    # Node 8: x[92] - (x[16]+x[17]+x[18]+x[19]+x[20]) = 0
    ExaModels.@add_con(c, x[92] - x[16]-x[17]-x[18]-x[19]-x[20])
    # Node 9: x[93] - x[21] = 0
    ExaModels.@add_con(c, x[93] - x[21])
    # Node 10: x[94] - x[22] = 0
    ExaModels.@add_con(c, x[94] - x[22])
    # Node 11: x[95] - (x[23]+x[24]+x[25]+x[26]) = 0
    ExaModels.@add_con(c, x[95] - x[23]-x[24]-x[25]-x[26])
    # Node 12: x[96] - (x[27]+x[28]) = 0
    ExaModels.@add_con(c, x[96] - x[27]-x[28])
    # Node 13: x[97] - (x[29]+x[30]) = 0
    ExaModels.@add_con(c, x[97] - x[29]-x[30])
    # Node 14: x[98] - x[31] = 0
    ExaModels.@add_con(c, x[98] - x[31])
    # Node 15: x[99] - (x[32]+x[33]) = 0
    ExaModels.@add_con(c, x[99] - x[32]-x[33])
    # Node 16: x[100] - x[34] = 0
    ExaModels.@add_con(c, x[100] - x[34])
    # Node 17: x[101] - x[35] = 0
    ExaModels.@add_con(c, x[101] - x[35])
    # Node 18: x[102] - x[36] = 0
    ExaModels.@add_con(c, x[102] - x[36])
    # Node 19: x[103] - x[37] = 0
    ExaModels.@add_con(c, x[103] - x[37])
    # Node 20: x[104] - (x[38]+x[39]+x[40]+x[41]+x[42]) = 0
    ExaModels.@add_con(c, x[104] - x[38]-x[39]-x[40]-x[41]-x[42])
    # Node 21: x[105] - (x[43]+x[44]+x[45]+x[46]+x[47]+x[48]+x[49]) = 0
    ExaModels.@add_con(c, x[105] - x[43]-x[44]-x[45]-x[46]-x[47]-x[48]-x[49])
    # Node 22: x[106] - (x[50]+x[51]+x[52]+x[53]+x[54]+x[55]+x[56]) = 0
    ExaModels.@add_con(c, x[106] - x[50]-x[51]-x[52]-x[53]-x[54]-x[55]-x[56])
    # Node 23: x[107] - (x[57]+x[58]+x[59]+x[60]+x[61]+x[62]) = 0
    ExaModels.@add_con(c, x[107] - x[57]-x[58]-x[59]-x[60]-x[61]-x[62])
    # Node 24: x[108] - x[63] = 0
    ExaModels.@add_con(c, x[108] - x[63])
    # Node 25: x[109] - x[64] = 0
    ExaModels.@add_con(c, x[109] - x[64])
    # Node 26: x[110] - x[65] = 0
    ExaModels.@add_con(c, x[110] - x[65])
    # Node 27: x[111] - x[66] = 0
    ExaModels.@add_con(c, x[111] - x[66])
    # Node 28: x[112] - x[67] = 0
    ExaModels.@add_con(c, x[112] - x[67])
    # Node 29: x[113] - x[68] = 0
    ExaModels.@add_con(c, x[113] - x[68])
    # Node 30: x[114] - x[69] = 0
    ExaModels.@add_con(c, x[114] - x[69])
    # Node 31: x[115] - (x[70]+x[71]+x[72]+x[73]+x[74]+x[75]+x[76]+x[77]+x[78]) = 0
    ExaModels.@add_con(c, x[115] - x[70]-x[71]-x[72]-x[73]-x[74]-x[75]-x[76]-x[77]-x[78])
    # Node 32: x[116] - (x[79]+x[80]+x[81]+x[82]+x[83]+x[84]) = 0
    ExaModels.@add_con(c, x[116] - x[79]-x[80]-x[81]-x[82]-x[83]-x[84])
    # Node 33: x[8] - x[85] = 0
    ExaModels.@add_con(c, x[8] - x[85])
    # Node 34: x[7] - x[86] = 0
    ExaModels.@add_con(c, x[7] - x[86])
    # Node 35: x[6] - x[87] = 0
    ExaModels.@add_con(c, x[6] - x[87])
    # Node 36: x[5] - x[88] = 0
    ExaModels.@add_con(c, x[5] - x[88])
    # Node 37: (x[12]+x[15]+x[20]) - x[89] = 0
    ExaModels.@add_con(c, x[12]+x[15]+x[20] - x[89])
    # Node 38: (x[14]+x[19]+x[21]+x[26]+x[84]) - x[90] = 0
    ExaModels.@add_con(c, x[14]+x[19]+x[21]+x[26]+x[84] - x[90])
    # Node 39: (x[11]+x[18]+x[25]) - x[91] = 0
    ExaModels.@add_con(c, x[11]+x[18]+x[25] - x[91])
    # Node 40: (x[10]+x[13]+x[22]+x[24]+x[83]) - x[92] = 0
    ExaModels.@add_con(c, x[10]+x[13]+x[22]+x[24]+x[83] - x[92])
    # Node 41: (x[42]+x[49]+x[56]+x[62]) - x[93] = 0
    ExaModels.@add_con(c, x[42]+x[49]+x[56]+x[62] - x[93])
    # Node 42: (x[48]+x[55]+x[61]) - x[94] = 0
    ExaModels.@add_con(c, x[48]+x[55]+x[61] - x[94])
    # Node 43: (x[41]+x[47]+x[54]) - x[95] = 0
    ExaModels.@add_con(c, x[41]+x[47]+x[54] - x[95])
    # Node 44: (x[1]+x[40]+x[46]+x[53]+x[60]) - x[96] = 0
    ExaModels.@add_con(c, x[1]+x[40]+x[46]+x[53]+x[60] - x[96])
    # Node 45: (x[2]+x[39]+x[45]+x[52]+x[59]) - x[97] = 0
    ExaModels.@add_con(c, x[2]+x[39]+x[45]+x[52]+x[59] - x[97])
    # Node 46: (x[3]+x[44]+x[51]+x[58]) - x[98] = 0
    ExaModels.@add_con(c, x[3]+x[44]+x[51]+x[58] - x[98])
    # Node 47: (x[4]+x[38]+x[43]+x[50]+x[57]) - x[99] = 0
    ExaModels.@add_con(c, x[4]+x[38]+x[43]+x[50]+x[57] - x[99])
    # Node 48: (x[28]+x[78]) - x[100] = 0
    ExaModels.@add_con(c, x[28]+x[78] - x[100])
    # Node 49: (x[30]+x[77]) - x[101] = 0
    ExaModels.@add_con(c, x[30]+x[77] - x[101])
    # Node 50: (x[31]+x[76]) - x[102] = 0
    ExaModels.@add_con(c, x[31]+x[76] - x[102])
    # Node 51: (x[33]+x[75]) - x[103] = 0
    ExaModels.@add_con(c, x[33]+x[75] - x[103])
    # Node 52: (x[34]+x[63]) - x[104] = 0
    ExaModels.@add_con(c, x[34]+x[63] - x[104])
    # Node 53: (x[35]+x[64]) - x[105] = 0
    ExaModels.@add_con(c, x[35]+x[64] - x[105])
    # Node 54: (x[36]+x[65]) - x[106] = 0
    ExaModels.@add_con(c, x[36]+x[65] - x[106])
    # Node 55: (x[37]+x[66]) - x[107] = 0
    ExaModels.@add_con(c, x[37]+x[66] - x[107])
    # Node 56: (x[74]+x[82]) - x[108] = 0
    ExaModels.@add_con(c, x[74]+x[82] - x[108])
    # Node 57: (x[73]+x[81]) - x[109] = 0
    ExaModels.@add_con(c, x[73]+x[81] - x[109])
    # Node 58: (x[72]+x[80]) - x[110] = 0
    ExaModels.@add_con(c, x[72]+x[80] - x[110])
    # Node 59: x[79] - x[111] = 0
    ExaModels.@add_con(c, x[79] - x[111])
    # Node 60: (x[27]+x[71]) - x[112] = 0
    ExaModels.@add_con(c, x[27]+x[71] - x[112])
    # Node 61: (x[29]+x[70]) - x[113] = 0
    ExaModels.@add_con(c, x[29]+x[70] - x[113])
    # Node 62: x[32] - x[114] = 0
    ExaModels.@add_con(c, x[32] - x[114])
    # Node 63: x[17] - x[115] = 0
    ExaModels.@add_con(c, x[17] - x[115])
    # Node 64: (x[9]+x[16]+x[23]+x[67]+x[68]+x[69]) - (x[116]+x[117]) = 0
    ExaModels.@add_con(c, x[9]+x[16]+x[23]+x[67]+x[68]+x[69] - x[116]-x[117])

    return ExaModels.ExaModel(c; kwargs...)
end
