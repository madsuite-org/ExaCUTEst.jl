function heart6_model(; T = Float64, backend = nothing, kwargs...)
    # Variables: a=x[1], c=x[2], t=x[3], u=x[4], v=x[5], w=x[6]
    Mx = T(-0.816)
    My = T(-0.017)
    sA = T(-1.826)
    sB = T(-0.754)
    sC = T(-4.839)
    sD = T(-3.259)
    sE = T(-14.023)
    sF = T(15.467)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 6; lvar = fill(T(-Inf), 6), uvar = fill(T(Inf), 6),
                       start = T[0, 0, 1, 1, 1, 1])

    # G1: E1+E2-E3-E4 = sA
    ExaModels.@add_con(c,
        x[3]*x[1] + x[4]*(Mx - x[1]) - x[5]*x[2] - x[6]*(My - x[2]) - sA)

    # G2: E5+E6+E7+E8 = sB
    ExaModels.@add_con(c,
        x[5]*x[1] + x[6]*(Mx - x[1]) + x[3]*x[2] + x[4]*(My - x[2]) - sB)

    # G3: E9 - 2*E10 + E11 - 2*E12 = sC
    ExaModels.@add_con(c,
        x[1]*(x[3]^2 - x[5]^2) - T(2)*x[2]*x[3]*x[5]
        + (Mx - x[1])*(x[4]^2 - x[6]^2) - T(2)*(My - x[2])*x[4]*x[6] - sC)

    # G4: E13 + 2*E14 + E15 + 2*E16 = sD
    ExaModels.@add_con(c,
        x[2]*(x[3]^2 - x[5]^2) + T(2)*x[1]*x[3]*x[5]
        + (My - x[2])*(x[4]^2 - x[6]^2) + T(2)*(Mx - x[1])*x[4]*x[6] - sD)

    # G5: E17+E18+E19+E20 = sE
    ExaModels.@add_con(c,
        x[1]*x[3]*(x[3]^2 - T(3)*x[5]^2) + x[2]*x[5]*(x[5]^2 - T(3)*x[3]^2)
        + (Mx - x[1])*x[4]*(x[4]^2 - T(3)*x[6]^2)
        + (My - x[2])*x[6]*(x[6]^2 - T(3)*x[4]^2) - sE)

    # G6: E21 - E22 + E23 - E24 = sF
    ExaModels.@add_con(c,
        x[2]*x[3]*(x[3]^2 - T(3)*x[5]^2) - x[1]*x[5]*(x[5]^2 - T(3)*x[3]^2)
        + (My - x[2])*x[4]*(x[4]^2 - T(3)*x[6]^2)
        - (Mx - x[1])*x[6]*(x[6]^2 - T(3)*x[4]^2) - sF)

    return ExaModels.ExaModel(c; kwargs...)
end
