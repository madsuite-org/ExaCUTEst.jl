# DECONVNE: Deconvolution problem (nonlinear equation version)
# FR DEFAULT: all vars free (lvar=-Inf), except C(K<=0) fixed at 0.
# 40 equality constraints: sum_{I=1}^{LGSG} SG(I)*C(K-I+1) = TR(K) for K=1..40
# CUTEst variable ordering (determined empirically):
#   SG(10)           → position 1   (free)
#   SG(11)           → position 2   (free)
#   C(-11..-2)       → positions 3..12  (fixed at 0)
#   C(1..40)         → positions 13..52 (free)
#   SG(1..9)         → positions 53..61 (free)
#   C(-1)            → position 62  (fixed at 0)
#   C(0)             → position 63  (fixed at 0)
function deconvne_model(; T = Float64, backend = nothing, kwargs...)
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

    NTOT = 63

    # CUTEst variable ordering:
    #   sg_idx(i) = i<=9 ? i+52 : i-9   (SG(1..9)→53..61, SG(10)→1, SG(11)→2)
    #   c_idx(k):  k>=1 → k+12, k=-11..-2 → k+14, k=-1 → 62, k=0 → 63
    sg_idx(i) = i <= 9 ? i + 52 : i - 9
    function c_idx(k)
        k >= 1  && return k + 12   # C(1..40)   → 13..52
        k == 0  && return 63       # C(0)        → 63 (fixed)
        k == -1 && return 62       # C(-1)       → 62 (fixed)
        return k + 14              # C(-11..-2)  → 3..12 (fixed)
    end

    lvar = fill(T(-Inf), NTOT)
    uvar = fill(T(Inf), NTOT)
    start_vals = zeros(T, NTOT)

    # Fix C(K) for K=-11..0 (lvar=uvar=0)
    for k in -LGSG:0
        idx = c_idx(k)
        lvar[idx] = T(0); uvar[idx] = T(0)
    end
    # SG start values
    for i in 1:LGSG
        start_vals[sg_idx(i)] = SSG[i]
    end

    ec = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(ec, x, NTOT; lvar = lvar, uvar = uvar, start = start_vals)

    # Equality constraints: only include I=1..min(K,LGSG) active terms (IDX.LE.0 → 0)
    for K in 1:LGTR
        nterms = min(K, LGSG)
        data_K = [(si=Int32(sg_idx(I)), ci=Int32(c_idx(K-I+1))) for I in 1:nterms]
        trK = TR[K]
        ExaModels.@add_con(ec, sum(x[d.si]*x[d.ci] for d in data_K) - trK)
    end

    return ExaModels.ExaModel(ec; kwargs...)
end
