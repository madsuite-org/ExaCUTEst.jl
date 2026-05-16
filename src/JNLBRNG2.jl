# JNLBRNG2: Journal bearing quadratic problem (excentricity=0.5), PT=PY=100
# Same as JNLBRNG1 but with EX=0.5 instead of 0.1
function jnlbrng2_model(PT = 100, PY = 100; T = Float64, backend = nothing, kwargs...)
    EX = T(0.5)
    atan1 = atan(T(1))
    PI4 = atan1
    LT = T(8) * PI4
    LY = T(20)

    PTm1 = PT - 1
    PYm1 = PY - 1
    HT = LT / T(PTm1)
    HY = LY / T(PYm1)
    HT_HY = HT / HY
    HY_HT = HY / HT
    CLINC = -EX * HT * HY

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

    ExaModels.@add_par(c, COEFF_R, COEFF_row)

    ExaModels.@add_obj(c,
        COEFF_R[ri] * x[(ri + 1 - 1) * PY + j]
        for ri in 1:(PT-2) for j in 2:(PY-1))

    all_la_ht2 = T[(T(2) * W[i] + W[i+1]) / T(6) * HY_HT for i in 1:(PT-1)]
    all_la_hy2 = T[(T(2) * W[i] + W[i+1]) / T(6) * HT_HY for i in 1:(PT-1)]
    all_mu_ht2 = T[(T(2) * W[i] + W[i-1]) / T(6) * HY_HT for i in 2:PT]
    all_mu_hy2 = T[(T(2) * W[i] + W[i-1]) / T(6) * HT_HY for i in 2:PT]

    ExaModels.@add_par(c, ALL_LA_T2, all_la_ht2)
    ExaModels.@add_par(c, ALL_LA_Y2, all_la_hy2)
    ExaModels.@add_par(c, ALL_MU_T2, all_mu_ht2)
    ExaModels.@add_par(c, ALL_MU_Y2, all_mu_hy2)

    ExaModels.@add_obj(c,
        ALL_LA_T2[ri] * (x[ri * PY + j] - x[(ri - 1) * PY + j])^2 +
        ALL_LA_Y2[ri] * (x[(ri - 1) * PY + j + 1] - x[(ri - 1) * PY + j])^2
        for ri in 1:(PT-1) for j in 1:(PY-1))

    ExaModels.@add_obj(c,
        ALL_MU_T2[ri] * (x[(ri - 2) * PY + j] - x[(ri - 1) * PY + j])^2 +
        ALL_MU_Y2[ri] * (x[(ri - 1) * PY + j - 1] - x[(ri - 1) * PY + j])^2
        for ri in 2:PT for j in 2:PY)

    return ExaModels.ExaModel(c; kwargs...)
end
