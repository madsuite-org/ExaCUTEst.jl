# JNLBRNGB: Journal bearing quadratic problem (excentricity=0.5), alternate formulation
# Same as JNLBRNGA but EX=0.5
function jnlbrngb_model(PT = 100, PY = 100; T = Float64, backend = nothing, kwargs...)
    EX = T(0.5)
    LT = T(6.2831853)
    LY = T(20)

    PTm1 = PT - 1
    PYm1 = PY - 1
    HT = LT / T(PTm1)
    HY = LY / T(PYm1)
    HT_HY = HT / HY
    HY_HT = HY / HT
    CLINC = -EX * HT * HY
    COEFF_INV = T(0.0833333333)

    nvar = PT * PY

    lvar = zeros(T, nvar)
    uvar = fill(T(Inf), nvar)
    start = zeros(T, nvar)

    for j in 1:PY
        uvar[(1 - 1) * PY + j] = T(0)
        uvar[(PT - 1) * PY + j] = T(0)
    end
    for i in 2:(PT-1)
        uvar[(i - 1) * PY + 1] = T(0)
        uvar[(i - 1) * PY + PY] = T(0)
    end

    XI = T[(i - 1) * HT for i in 1:PT]
    W = T[(T(1) + EX * cos(XI[i]))^3 for i in 1:PT]
    SIN_XI = T[sin(XI[i]) for i in 1:PT]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    interior_rows = 2:(PT-1)
    COEFF_row = T[CLINC * SIN_XI[i] for i in interior_rows]
    MU_HT2 = T[(T(2) * W[i] + W[i+1]) * COEFF_INV * HY_HT for i in interior_rows]
    MU_HY2 = T[(T(2) * W[i] + W[i+1]) * COEFF_INV * HT_HY for i in interior_rows]
    LA_HT2 = T[(T(2) * W[i] + W[i-1]) * COEFF_INV * HY_HT for i in interior_rows]
    LA_HY2 = T[(T(2) * W[i] + W[i-1]) * COEFF_INV * HT_HY for i in interior_rows]

    ExaModels.@add_par(c, COEFF_R, COEFF_row)
    ExaModels.@add_par(c, MU_T2, MU_HT2)
    ExaModels.@add_par(c, MU_Y2, MU_HY2)
    ExaModels.@add_par(c, LA_T2, LA_HT2)
    ExaModels.@add_par(c, LA_Y2, LA_HY2)

    ExaModels.@add_obj(c,
        COEFF_R[ri] * x[ri * PY + j] +
        MU_T2[ri] * (x[(ri + 1) * PY + j] - x[ri * PY + j])^2 +
        MU_Y2[ri] * (x[ri * PY + j + 1] - x[ri * PY + j])^2 +
        LA_T2[ri] * (x[(ri - 1) * PY + j] - x[ri * PY + j])^2 +
        LA_Y2[ri] * (x[ri * PY + j - 1] - x[ri * PY + j])^2
        for ri in 1:(PT-2) for j in 2:(PY-1))

    return ExaModels.ExaModel(c; kwargs...)
end
