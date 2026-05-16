# DECONVC: Deconvolution problem (nonlinearly constrained version)
# Same structure as DECONVB but with constraint sum_{I=1}^{LGSG} SG(I)^2 = PIC=12.35
# CUTEst variable ordering (different from declaration order):
#   SG(1..11)         → positions 1..11
#   C(0)              → position 12  (fixed at 0)
#   C(1..40)          → positions 13..52
#   C(-11..-1)        → positions 53..63 (fixed at 0)
function deconvc_model(; T = Float64, backend = nothing, kwargs...)
    LGSG = 11
    LGTR = 40
    PIC  = T(12.35)

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

    NTOT = 63  # 11 SG + 1 C(0) + 40 C(1..40) + 11 C(-11..-1)

    # CUTEst variable ordering:
    #   sg_idx(i) = i              (i=1..11)
    #   c_idx(k)  = k+12           (k=1..40)  free
    #   c_idx(0)  = 12             fixed at 0
    #   c_idx(k)  = k+64           (k=-11..-1) fixed at 0
    sg_idx(i) = i
    c_idx(k)  = k >= 1 ? k + 12 : (k == 0 ? 12 : k + 64)

    lvar = zeros(T, NTOT)
    uvar = fill(T(Inf), NTOT)
    start_vals = zeros(T, NTOT)

    # Fix C(K) for K=-11..0 (lvar=uvar=0)
    uvar[c_idx(0)] = T(0)
    for k in -LGSG:-1
        uvar[c_idx(k)] = T(0)
    end
    # SG start values
    for i in 1:LGSG
        start_vals[sg_idx(i)] = SSG[i]
    end

    ec = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(ec, x, NTOT; lvar = lvar, uvar = uvar, start = start_vals)

    # Objective: only include I=1..min(K,LGSG) active terms (IDX.LE.0 → SCAL=0 in SIF)
    for K in 1:LGTR
        nterms = min(K, LGSG)
        data_K = [(si=Int32(sg_idx(I)), ci=Int32(c_idx(K-I+1))) for I in 1:nterms]
        trK = TR[K]
        ExaModels.@add_obj(ec, (sum(x[d.si]*x[d.ci] for d in data_K) - trK)^2)
    end

    # Constraint ENERG: sum_{I=1}^{LGSG} SG(I)^2 - PIC = 0
    sg_data = [(si=Int32(sg_idx(i)),) for i in 1:LGSG]
    ExaModels.@add_con(ec, sum(x[d.si]^2 for d in sg_data) - PIC)

    return ExaModels.ExaModel(ec; kwargs...)
end
