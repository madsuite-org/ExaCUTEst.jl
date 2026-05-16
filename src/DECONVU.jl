# DECONVU: Deconvolution problem (unconstrained version)
# Same as DECONVB but all variables free (FR DEFAULT), except C(K<=0) fixed at 0.
# Objective: sum_{K=1}^{LGTR} (sum_{I=1}^{LGSG} SG(I)*C(K-I+1) - TR(K))^2
function deconvu_model(; T = Float64, backend = nothing, kwargs...)
    LGSG = 11
    LGTR = 40

    TR = T[0.0, 0.0, 1.600000e-03, 5.400000e-03, 7.020000e-02,
           0.1876, 0.3320, 0.7640, 0.9320, 0.8120,
           0.3464, 0.2064, 8.300000e-02, 3.400000e-02, 6.179999e-02,
           1.2, 1.8, 2.4, 9.0, 2.4,
           1.8010, 1.3250, 7.620000e-02, 0.2104, 0.2680,
           0.5520, 0.9960, 0.3600, 0.2400, 0.1510,
           2.480000e-02, 0.2432, 0.3602, 0.4800, 1.8000,
           0.4800, 0.3600, 0.2640, 6.000000e-03, 6.000000e-03]

    SSG = T[1.000000e-02, 2.000000e-02, 0.4, 0.6, 0.8, 3.0, 0.8, 0.6,
            0.4400, 1.000000e-02, 1.000000e-02]

    NC = LGTR + LGSG + 1  # = 52
    c_idx(k) = k + LGSG + 1
    sg_idx(i) = NC + i
    NTOT = NC + LGSG  # = 63

    lvar = fill(T(-Inf), NTOT)
    uvar = fill(T(Inf), NTOT)
    start_vals = zeros(T, NTOT)

    # C(K) for K = -LGSG..0: fixed at 0 (XX)
    for k in -LGSG:0
        idx = c_idx(k)
        lvar[idx] = T(0); uvar[idx] = T(0)
    end
    # SG start
    for i in 1:LGSG
        start_vals[sg_idx(i)] = SSG[i]
    end

    ec = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(ec, x, NTOT; lvar = lvar, uvar = uvar, start = start_vals)

    # Only include I=1..min(K,LGSG) active terms per K (IDX.LE.0 → SCAL=0 in SIF)
    for K in 1:LGTR
        nterms = min(K, LGSG)
        data_K = [(si=Int32(sg_idx(I)), ci=Int32(c_idx(K-I+1))) for I in 1:nterms]
        trK = TR[K]
        ExaModels.@add_obj(ec, (sum(x[d.si]*x[d.ci] for d in data_K) - trK)^2)
    end

    return ExaModels.ExaModel(ec; kwargs...)
end
