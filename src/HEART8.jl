function heart8_model(; T = Float64, backend = nothing, kwargs...)
    # Variables: a=x[1], b=x[2], c=x[3], d=x[4], t=x[5], u=x[6], v=x[7], w=x[8]
    Mx = T(-0.69)
    My = T(-0.044)
    sA = T(-1.57)
    sB = T(-1.31)
    sC = T(-2.65)
    sD = T(2.0)
    sE = T(-12.6)
    sF = T(9.48)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 8; lvar = fill(T(-Inf), 8), uvar = fill(T(Inf), 8),
                       start = T[0, 1, 0, 1, 1, 1, 1, 1])

    # G1: a + b = sum_Mx
    ExaModels.@add_con(c, x[1] + x[2] - Mx)

    # G2: c + d = sum_My
    ExaModels.@add_con(c, x[3] + x[4] - My)

    # G3: E1+E2-E3-E4 = sA
    ExaModels.@add_con(c,
        x[5]*x[1] + x[6]*x[2] - x[7]*x[3] - x[8]*x[4] - sA)

    # G4: E5+E6+E7+E8 = sB
    ExaModels.@add_con(c,
        x[7]*x[1] + x[8]*x[2] + x[5]*x[3] + x[6]*x[4] - sB)

    # G5: E9 - 2*E10 + E11 - 2*E12 = sC
    ExaModels.@add_con(c,
        x[1]*(x[5]^2 - x[7]^2) - T(2)*x[3]*x[5]*x[7]
        + x[2]*(x[6]^2 - x[8]^2) - T(2)*x[4]*x[6]*x[8] - sC)

    # G6: E13 + 2*E14 + E15 + 2*E16 = sD
    ExaModels.@add_con(c,
        x[3]*(x[5]^2 - x[7]^2) + T(2)*x[1]*x[5]*x[7]
        + x[4]*(x[6]^2 - x[8]^2) + T(2)*x[2]*x[6]*x[8] - sD)

    # G7: E17+E18+E19+E20 = sE
    ExaModels.@add_con(c,
        x[1]*x[5]*(x[5]^2 - T(3)*x[7]^2) + x[3]*x[7]*(x[7]^2 - T(3)*x[5]^2)
        + x[2]*x[6]*(x[6]^2 - T(3)*x[8]^2) + x[4]*x[8]*(x[8]^2 - T(3)*x[6]^2) - sE)

    # G8: E21 - E22 + E23 - E24 = sF
    ExaModels.@add_con(c,
        x[3]*x[5]*(x[5]^2 - T(3)*x[7]^2) - x[1]*x[7]*(x[7]^2 - T(3)*x[5]^2)
        + x[4]*x[6]*(x[6]^2 - T(3)*x[8]^2) - x[2]*x[8]*(x[8]^2 - T(3)*x[6]^2) - sF)

    return ExaModels.ExaModel(c; kwargs...)
end
