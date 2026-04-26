# HIMMELP6: Himmelblau's nonlinear problem (case V, 5 constraints)
# Source: Pshenichnyj (1994), SIF input: Toint, December 1994
# 2 variables, 5 constraints
function himmelp6_model(; T = Float64, backend = nothing, kwargs...)
    nB2 = T(3.8112755343)
    nB6 = T(6.8306567613)
    B1  = T(75.1963666677)
    B3  = T(0.1269366345)
    B4  = T(-0.0020567665)
    B5  = T(1.0345e-5)
    B7  = T(0.0302344793)
    B8  = T(-0.0012813448)
    B9  = T(3.52599e-5)
    B10 = T(-2.266e-7)
    B11 = T(0.2564581253)
    B12 = T(-0.003460403)
    B13 = T(1.35139e-5)
    B14 = T(-28.1064434908)
    B15 = T(-5.2375e-6)
    B16 = T(-6.3e-9)
    B17 = T(7e-10)
    B18 = T(0.0003405462)
    B19 = T(-1.6638e-6)
    B20 = T(-2.8673112392)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2;
        lvar = T[0, 0], uvar = T[75, 65], start = T[68.8, 31.2])

    ExaModels.@add_obj(c,
        nB2*x[1] + nB6*x[2] - B1 - (
            B3*x[1]^2 + B4*x[1]^3 + B5*x[1]^4
            + x[2]*(B7*x[1] + B8*x[1]^2 + B9*x[1]^3 + B10*x[1]^4)
            + B11*x[2]^2 + B12*x[2]^3 + B13*x[2]^4
            + B14/(T(1) + x[2])
            + (B18*x[1] + B15*x[1]^2 + B16*x[1]^3)*x[2]^2
            + (B17*x[1]^3 + B19*x[1])*x[2]^3
            + B20*exp(T(0.0005)*x[1]*x[2])
        )
    )

    # C5: X1*X2 - 700 >= 0
    ExaModels.@add_con(c, x[1]*x[2] - T(700); lcon = T(0), ucon = T(Inf))
    # C8: -X2 + 0.008*X1^2 <= 0
    ExaModels.@add_con(c, -x[2] + T(0.008)*x[1]^2; lcon = T(-Inf), ucon = T(0))
    # C9: 5*X1 + 100*X2 - X2^2 - 2775 <= 0
    ExaModels.@add_con(c, T(5)*x[1] + T(100)*x[2] - x[2]^2 - T(2775);
        lcon = T(-Inf), ucon = T(0))
    # C11: X1 - 1.5*X2 - 22.5 <= 0 (purely linear)
    ExaModels.@add_con(c, x[1] - T(1.5)*x[2] - T(22.5); lcon = T(-Inf), ucon = T(0))
    # C12: -X1 + 0.16*X2 - 41.4 <= 0 (purely linear)
    ExaModels.@add_con(c, -x[1] + T(0.16)*x[2] - T(41.4); lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
