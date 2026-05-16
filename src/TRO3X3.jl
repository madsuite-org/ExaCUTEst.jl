function tro3x3_model(; T = Float64, backend = nothing, kwargs...)
    # TRO3X3: Minimum-compliance truss topology problem
    # classification LOR2-RN-30-13
    # Variables: x[1..18] = bar areas X(1..18), x[19..30] = displacements U(1..12)
    # Objective: minimize U(8) = x[26]

    one_over_M = T(1.0/18.0)
    lvar = vcat(fill(T(0), 18), fill(T(-Inf), 12))
    uvar = fill(T(Inf), 30)
    x0   = vcat(fill(one_over_M, 18), fill(T(0), 12))

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 30; lvar = lvar, uvar = uvar, start = x0)

    # Objective: minimize U(8) = x[26]
    ExaModels.@add_obj(c, x[26])

    # BALANCE: sum X(1..18) <= 1
    ExaModels.@add_con(c, x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]+
                           x[11]+x[12]+x[13]+x[14]+x[15]+x[16]+x[17]+x[18] - T(1);
                       lcon = T(-Inf), ucon = T(0))

    # C(1) = 0
    ExaModels.@add_con(c, T(4)*x[1]*x[19]+x[4]*x[19]-x[4]*x[20]+T(4)*x[9]*x[19]-T(4)*x[9]*x[25]
                        +x[10]*x[19]+x[10]*x[20]-x[10]*x[27]-x[10]*x[28])

    # C(2) = 0
    ExaModels.@add_con(c, T(4)*x[3]*x[20]-T(4)*x[3]*x[22]-x[4]*x[19]+x[4]*x[20]+x[10]*x[19]
                        +x[10]*x[20]-x[10]*x[27]-x[10]*x[28])

    # C(3) = 0
    ExaModels.@add_con(c, x[2]*x[21]+x[2]*x[22]+T(4)*x[5]*x[21]+x[8]*x[21]-x[8]*x[22]+x[12]*x[21]
                        -x[12]*x[22]-x[12]*x[25]+x[12]*x[26]+T(4)*x[13]*x[21]-T(4)*x[13]*x[27]
                        +x[14]*x[21]+x[14]*x[22]-x[14]*x[29]-x[14]*x[30])

    # C(4) = 0
    ExaModels.@add_con(c, x[2]*x[21]+x[2]*x[22]-T(4)*x[3]*x[20]+T(4)*x[3]*x[22]+T(4)*x[7]*x[22]
                        -T(4)*x[7]*x[24]-x[8]*x[21]+x[8]*x[22]-x[12]*x[21]+x[12]*x[22]+x[12]*x[25]
                        -x[12]*x[26]+x[14]*x[21]+x[14]*x[22]-x[14]*x[29]-x[14]*x[30])

    # C(5) = 0
    ExaModels.@add_con(c, x[6]*x[23]+x[6]*x[24]+x[16]*x[23]-x[16]*x[24]-x[16]*x[27]+x[16]*x[28]
                        +T(4)*x[17]*x[23]+T(4)*x[18]*x[23]-T(4)*x[18]*x[29])

    # C(6) = 0
    ExaModels.@add_con(c, x[6]*x[23]+x[6]*x[24]-T(4)*x[7]*x[22]+T(4)*x[7]*x[24]-x[16]*x[23]
                        +x[16]*x[24]+x[16]*x[27]-x[16]*x[28])

    # C(7) = 0
    ExaModels.@add_con(c, -T(4)*x[9]*x[19]+T(4)*x[9]*x[25]-x[12]*x[21]+x[12]*x[22]+x[12]*x[25]
                        -x[12]*x[26])

    # C(8) = 1
    ExaModels.@add_con(c, T(4)*x[11]*x[26]-T(4)*x[11]*x[28]+x[12]*x[21]-x[12]*x[22]-x[12]*x[25]
                        +x[12]*x[26]-T(1))

    # C(9) = 0
    ExaModels.@add_con(c, -x[10]*x[19]-x[10]*x[20]+x[10]*x[27]+x[10]*x[28]-T(4)*x[13]*x[21]
                        +T(4)*x[13]*x[27]-x[16]*x[23]+x[16]*x[24]+x[16]*x[27]-x[16]*x[28])

    # C(10) = 0
    ExaModels.@add_con(c, -x[10]*x[19]-x[10]*x[20]+x[10]*x[27]+x[10]*x[28]-T(4)*x[11]*x[26]
                        +T(4)*x[11]*x[28]+T(4)*x[15]*x[28]-T(4)*x[15]*x[30]+x[16]*x[23]-x[16]*x[24]
                        -x[16]*x[27]+x[16]*x[28])

    # C(11) = 0
    ExaModels.@add_con(c, -x[14]*x[21]-x[14]*x[22]+x[14]*x[29]+x[14]*x[30]-T(4)*x[18]*x[23]
                        +T(4)*x[18]*x[29])

    # C(12) = 0
    ExaModels.@add_con(c, -x[14]*x[21]-x[14]*x[22]+x[14]*x[29]+x[14]*x[30]-T(4)*x[15]*x[28]
                        +T(4)*x[15]*x[30])

    return ExaModels.ExaModel(c; kwargs...)
end
