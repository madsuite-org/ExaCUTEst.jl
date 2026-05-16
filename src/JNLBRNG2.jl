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

    data_lin = [(ri = ri, idx = ri * PY + j) for ri in 1:(PT-2) for j in 2:(PY-1)]
    ExaModels.@add_obj(c, COEFF_R[d.ri] * x[d.idx] for d in data_lin)

    all_la_ht2 = T[(T(2) * W[i] + W[i+1]) / T(12) * HY_HT for i in 1:(PT-1)]
    all_la_hy2 = T[(T(2) * W[i] + W[i+1]) / T(12) * HT_HY for i in 1:(PT-1)]
    all_mu_ht2 = T[(T(2) * W[i] + W[i-1]) / T(12) * HY_HT for i in 2:PT]
    all_mu_hy2 = T[(T(2) * W[i] + W[i-1]) / T(12) * HT_HY for i in 2:PT]

    ExaModels.@add_par(c, ALL_LA_T2, all_la_ht2)
    ExaModels.@add_par(c, ALL_LA_Y2, all_la_hy2)
    ExaModels.@add_par(c, ALL_MU_T2, all_mu_ht2)
    ExaModels.@add_par(c, ALL_MU_Y2, all_mu_hy2)

    data_right = [(ri = ri, idx_ip = ri * PY + j, idx_ij = (ri - 1) * PY + j, idx_jp = (ri - 1) * PY + j + 1) for ri in 1:(PT-1) for j in 1:(PY-1)]
    ExaModels.@add_obj(c,
        ALL_LA_T2[d.ri] * (x[d.idx_ip] - x[d.idx_ij])^2 +
        ALL_LA_Y2[d.ri] * (x[d.idx_jp] - x[d.idx_ij])^2
        for d in data_right)

    data_left = [(mu_ri = ri - 1, idx_im = (ri - 2) * PY + j, idx_ij = (ri - 1) * PY + j, idx_jm = (ri - 1) * PY + j - 1) for ri in 2:PT for j in 2:PY]
    ExaModels.@add_obj(c,
        ALL_MU_T2[d.mu_ri] * (x[d.idx_im] - x[d.idx_ij])^2 +
        ALL_MU_Y2[d.mu_ri] * (x[d.idx_jm] - x[d.idx_ij])^2
        for d in data_left)

    return ExaModels.ExaModel(c; kwargs...)
end
