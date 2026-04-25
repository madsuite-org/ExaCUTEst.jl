# LUKVLE2: Chained Wood function with Broyden banded equality constraints
# Source: Problem 5.2, Luksan & Vlcek (1999), TR 767
# N=10000 variables, N/2-1=4999 equality constraints
# All variables free; start: odd→-2, even→1
#
# CUTEst variable ordering (sifdecoder reordering):
#   perm(m) = m           for m ∈ {4995..5005} ∪ {10000}  (identity window)
#   perm(m) = N - m       otherwise                         (reversed)
#
# Because N=10000 is even, perm preserves odd/even parity, so the alternating
# start vector sif_start(k) = isodd(k)?-2:1 maps to an identical CUTEst start.
#
# Objective groups (I=1..N/2-1=4999):
#   A(I): 100*(-x[perm(2I)]   + x[perm(2I-1)]^2)^2
#   B(I): (x[perm(2I-1)]  - 1)^2
#   C(I=1..5): 90*(-x[perm(2I+2)] + x[perm(2I+1)]^2)^2  (simple wood term)
#   C(I=6..4999): 90*v^2 where v = wood + Broyden(center=perm(I), band=perm(I-5..I+1))
#   D(I): (-x[perm(2I+1)] - 1)^2
#   E(I): 10*(x[perm(2I)] + x[perm(2I+2)] - 2)^2
#   F(I): (x[perm(2I)] - x[perm(2I-1)])^2 / 10
#
# Constraints (K=5000..9998, ci=K-4999=1..4999):
#   2*x[perm(K)] + sum_{J=K-5}^{K+1} x[perm(J)]
#   + 5*x[perm(K)]^3 + sum_{J=K-5}^{K+1} x[perm(J)]^2 + 1 = 0
function lukvle2_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert N == 10000 "LUKVLE2 uses a fixed sifdecoder variable ordering only valid for N=10000"
    N2m1 = N ÷ 2 - 1  # = 4999

    # Start vector: perm preserves parity so alternating -2/1 is identical in CUTEst order
    start = T[isodd(i) ? T(-2) : T(1) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # ─── Objective ───────────────────────────────────────────────────────────────
    # For I in the "fully reversed" regions (1..2496 and 2504..4999):
    #   perm(2I-1)=10001-2I, perm(2I)=10000-2I, perm(2I+1)=9999-2I, perm(2I+2)=9998-2I
    # (except perm(2*4999+2)=perm(10000)=10000, handled separately for E)

    # Groups A, B, D, F — reversed loop I=1..2496
    ExaModels.@add_obj(c, T(100) * (-x[10000-2i] + x[10001-2i]^2)^2 for i in 1:2496)
    ExaModels.@add_obj(c, (x[10001-2i] - T(1))^2 for i in 1:2496)
    ExaModels.@add_obj(c, (-x[9999-2i] - T(1))^2 for i in 1:2496)
    ExaModels.@add_obj(c, T(10) * (x[10000-2i] + x[9998-2i] - T(2))^2 for i in 1:2496)
    ExaModels.@add_obj(c, (x[10000-2i] - x[10001-2i])^2 / T(10) for i in 1:2496)

    # Groups A, B, D, F — reversed loop I=2504..4999
    ExaModels.@add_obj(c, T(100) * (-x[10000-2i] + x[10001-2i]^2)^2 for i in 2504:4999)
    ExaModels.@add_obj(c, (x[10001-2i] - T(1))^2 for i in 2504:4999)
    ExaModels.@add_obj(c, (-x[9999-2i] - T(1))^2 for i in 2504:4999)
    # E: reversed loop I=2504..4998 (I=4999 is special because perm(10000)=10000)
    ExaModels.@add_obj(c, T(10) * (x[10000-2i] + x[9998-2i] - T(2))^2 for i in 2504:4998)
    ExaModels.@add_obj(c, (x[10000-2i] - x[10001-2i])^2 / T(10) for i in 2504:4999)

    # E(I=4999): perm(9998)=2, perm(10000)=10000
    ExaModels.@add_obj(c, T(10) * (x[2] + x[10000] - T(2))^2)

    # Groups A,B,D,E,F — 7 individual boundary cases I=2497..2503
    # I=2497: perm(4993)=5007, perm(4994)=5006, perm(4995)=4995, perm(4996)=4996
    ExaModels.@add_obj(c, T(100) * (-x[5006] + x[5007]^2)^2)
    ExaModels.@add_obj(c, (x[5007] - T(1))^2)
    ExaModels.@add_obj(c, (-x[4995] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[5006] + x[4996] - T(2))^2)
    ExaModels.@add_obj(c, (x[5006] - x[5007])^2 / T(10))
    # I=2498: perm(4995)=4995, perm(4996)=4996, perm(4997)=4997, perm(4998)=4998
    ExaModels.@add_obj(c, T(100) * (-x[4996] + x[4995]^2)^2)
    ExaModels.@add_obj(c, (x[4995] - T(1))^2)
    ExaModels.@add_obj(c, (-x[4997] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[4996] + x[4998] - T(2))^2)
    ExaModels.@add_obj(c, (x[4996] - x[4995])^2 / T(10))
    # I=2499: perm(4997)=4997, perm(4998)=4998, perm(4999)=4999, perm(5000)=5000
    ExaModels.@add_obj(c, T(100) * (-x[4998] + x[4997]^2)^2)
    ExaModels.@add_obj(c, (x[4997] - T(1))^2)
    ExaModels.@add_obj(c, (-x[4999] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[4998] + x[5000] - T(2))^2)
    ExaModels.@add_obj(c, (x[4998] - x[4997])^2 / T(10))
    # I=2500: perm(4999)=4999, perm(5000)=5000, perm(5001)=5001, perm(5002)=5002
    ExaModels.@add_obj(c, T(100) * (-x[5000] + x[4999]^2)^2)
    ExaModels.@add_obj(c, (x[4999] - T(1))^2)
    ExaModels.@add_obj(c, (-x[5001] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[5000] + x[5002] - T(2))^2)
    ExaModels.@add_obj(c, (x[5000] - x[4999])^2 / T(10))
    # I=2501: perm(5001)=5001, perm(5002)=5002, perm(5003)=5003, perm(5004)=5004
    ExaModels.@add_obj(c, T(100) * (-x[5002] + x[5001]^2)^2)
    ExaModels.@add_obj(c, (x[5001] - T(1))^2)
    ExaModels.@add_obj(c, (-x[5003] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[5002] + x[5004] - T(2))^2)
    ExaModels.@add_obj(c, (x[5002] - x[5001])^2 / T(10))
    # I=2502: perm(5003)=5003, perm(5004)=5004, perm(5005)=5005, perm(5006)=4994
    ExaModels.@add_obj(c, T(100) * (-x[5004] + x[5003]^2)^2)
    ExaModels.@add_obj(c, (x[5003] - T(1))^2)
    ExaModels.@add_obj(c, (-x[5005] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[5004] + x[4994] - T(2))^2)
    ExaModels.@add_obj(c, (x[5004] - x[5003])^2 / T(10))
    # I=2503: perm(5005)=5005, perm(5006)=4994, perm(5007)=4993, perm(5008)=4992
    ExaModels.@add_obj(c, T(100) * (-x[4994] + x[5005]^2)^2)
    ExaModels.@add_obj(c, (x[5005] - T(1))^2)
    ExaModels.@add_obj(c, (-x[4993] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[4994] + x[4992] - T(2))^2)
    ExaModels.@add_obj(c, (x[4994] - x[5005])^2 / T(10))

    # Group C(I=1..5): simple wood term (no Broyden), fully reversed
    # perm(2I+1)=9999-2I, perm(2I+2)=9998-2I for I=1..5 (all < 4995)
    ExaModels.@add_obj(c, T(90) * (-x[9998-2i] + x[9999-2i]^2)^2 for i in 1:5)

    # Group C(I=6..4993): Broyden banded, fully reversed for I outside wood-identity window.
    # For I=6..2496 and I=2503..4993: perm(2I+1)=9999-2I and perm(2I+2)=9998-2I are reversed.
    # For I=2497..2502: 2I+1 or 2I+2 fall in identity window {4995..5005} → handled individually below.
    ExaModels.@add_obj(
        c,
        T(90) * (
            -x[9998-2i] + x[9999-2i]^2 +
            T(2) * x[10000-i] +
            (x[10005-i] + x[10004-i] + x[10003-i] + x[10002-i] + x[10001-i] + x[10000-i] + x[9999-i]) +
            T(5) * x[10000-i]^3 +
            (x[10005-i]^2 + x[10004-i]^2 + x[10003-i]^2 + x[10002-i]^2 + x[10001-i]^2 + x[10000-i]^2 + x[9999-i]^2) +
            T(1)
        )^2 for i in 6:2496
    )
    ExaModels.@add_obj(
        c,
        T(90) * (
            -x[9998-2i] + x[9999-2i]^2 +
            T(2) * x[10000-i] +
            (x[10005-i] + x[10004-i] + x[10003-i] + x[10002-i] + x[10001-i] + x[10000-i] + x[9999-i]) +
            T(5) * x[10000-i]^3 +
            (x[10005-i]^2 + x[10004-i]^2 + x[10003-i]^2 + x[10002-i]^2 + x[10001-i]^2 + x[10000-i]^2 + x[9999-i]^2) +
            T(1)
        )^2 for i in 2503:4993
    )

    # C(I=2497..2502): Broyden banded, center/band reversed but wood positions in identity window.
    # Center=x[10000-I], band=x[10005-I..9999-I]; wood=perm(2I+2) and perm(2I+1) (correct below).
    # I=2497: perm(4995)=4995, perm(4996)=4996; center=7503, band=7508..7502
    ExaModels.@add_obj(c, T(90)*(-x[4996]+x[4995]^2+T(2)*x[7503]+(x[7508]+x[7507]+x[7506]+x[7505]+x[7504]+x[7503]+x[7502])+T(5)*x[7503]^3+(x[7508]^2+x[7507]^2+x[7506]^2+x[7505]^2+x[7504]^2+x[7503]^2+x[7502]^2)+T(1))^2)
    # I=2498: perm(4997)=4997, perm(4998)=4998; center=7502, band=7507..7501
    ExaModels.@add_obj(c, T(90)*(-x[4998]+x[4997]^2+T(2)*x[7502]+(x[7507]+x[7506]+x[7505]+x[7504]+x[7503]+x[7502]+x[7501])+T(5)*x[7502]^3+(x[7507]^2+x[7506]^2+x[7505]^2+x[7504]^2+x[7503]^2+x[7502]^2+x[7501]^2)+T(1))^2)
    # I=2499: perm(4999)=4999, perm(5000)=5000; center=7501, band=7506..7500
    ExaModels.@add_obj(c, T(90)*(-x[5000]+x[4999]^2+T(2)*x[7501]+(x[7506]+x[7505]+x[7504]+x[7503]+x[7502]+x[7501]+x[7500])+T(5)*x[7501]^3+(x[7506]^2+x[7505]^2+x[7504]^2+x[7503]^2+x[7502]^2+x[7501]^2+x[7500]^2)+T(1))^2)
    # I=2500: perm(5001)=5001, perm(5002)=5002; center=7500, band=7505..7499
    ExaModels.@add_obj(c, T(90)*(-x[5002]+x[5001]^2+T(2)*x[7500]+(x[7505]+x[7504]+x[7503]+x[7502]+x[7501]+x[7500]+x[7499])+T(5)*x[7500]^3+(x[7505]^2+x[7504]^2+x[7503]^2+x[7502]^2+x[7501]^2+x[7500]^2+x[7499]^2)+T(1))^2)
    # I=2501: perm(5003)=5003, perm(5004)=5004; center=7499, band=7504..7498
    ExaModels.@add_obj(c, T(90)*(-x[5004]+x[5003]^2+T(2)*x[7499]+(x[7504]+x[7503]+x[7502]+x[7501]+x[7500]+x[7499]+x[7498])+T(5)*x[7499]^3+(x[7504]^2+x[7503]^2+x[7502]^2+x[7501]^2+x[7500]^2+x[7499]^2+x[7498]^2)+T(1))^2)
    # I=2502: perm(5005)=5005, perm(5006)=4994; center=7498, band=7503..7497
    ExaModels.@add_obj(c, T(90)*(-x[4994]+x[5005]^2+T(2)*x[7498]+(x[7503]+x[7502]+x[7501]+x[7500]+x[7499]+x[7498]+x[7497])+T(5)*x[7498]^3+(x[7503]^2+x[7502]^2+x[7501]^2+x[7500]^2+x[7499]^2+x[7498]^2+x[7497]^2)+T(1))^2)

    # Group C(I=4994..4999): 6 individual boundary cases
    # C(I=4994): wood=-x[10]+x[11]^2, center=x[5006], band includes x[4995] (identity)
    ExaModels.@add_obj(c, T(90) * (
        -x[10] + x[11]^2 + T(2)*x[5006] +
        (x[5011]+x[5010]+x[5009]+x[5008]+x[5007]+x[5006]+x[4995]) +
        T(5)*x[5006]^3 +
        (x[5011]^2+x[5010]^2+x[5009]^2+x[5008]^2+x[5007]^2+x[5006]^2+x[4995]^2) + T(1)
    )^2)
    # C(I=4995): wood=-x[8]+x[9]^2, center=x[4995], band includes x[4996] (identity)
    ExaModels.@add_obj(c, T(90) * (
        -x[8] + x[9]^2 + T(2)*x[4995] +
        (x[5010]+x[5009]+x[5008]+x[5007]+x[5006]+x[4995]+x[4996]) +
        T(5)*x[4995]^3 +
        (x[5010]^2+x[5009]^2+x[5008]^2+x[5007]^2+x[5006]^2+x[4995]^2+x[4996]^2) + T(1)
    )^2)
    # C(I=4996): wood=-x[6]+x[7]^2, center=x[4996]
    ExaModels.@add_obj(c, T(90) * (
        -x[6] + x[7]^2 + T(2)*x[4996] +
        (x[5009]+x[5008]+x[5007]+x[5006]+x[4995]+x[4996]+x[4997]) +
        T(5)*x[4996]^3 +
        (x[5009]^2+x[5008]^2+x[5007]^2+x[5006]^2+x[4995]^2+x[4996]^2+x[4997]^2) + T(1)
    )^2)
    # C(I=4997): wood=-x[4]+x[5]^2, center=x[4997]
    ExaModels.@add_obj(c, T(90) * (
        -x[4] + x[5]^2 + T(2)*x[4997] +
        (x[5008]+x[5007]+x[5006]+x[4995]+x[4996]+x[4997]+x[4998]) +
        T(5)*x[4997]^3 +
        (x[5008]^2+x[5007]^2+x[5006]^2+x[4995]^2+x[4996]^2+x[4997]^2+x[4998]^2) + T(1)
    )^2)
    # C(I=4998): wood=-x[2]+x[3]^2, center=x[4998]
    ExaModels.@add_obj(c, T(90) * (
        -x[2] + x[3]^2 + T(2)*x[4998] +
        (x[5007]+x[5006]+x[4995]+x[4996]+x[4997]+x[4998]+x[4999]) +
        T(5)*x[4998]^3 +
        (x[5007]^2+x[5006]^2+x[4995]^2+x[4996]^2+x[4997]^2+x[4998]^2+x[4999]^2) + T(1)
    )^2)
    # C(I=4999): wood=-x[10000]+x[1]^2, center=x[4999], band includes x[5000] (identity)
    ExaModels.@add_obj(c, T(90) * (
        -x[10000] + x[1]^2 + T(2)*x[4999] +
        (x[5006]+x[4995]+x[4996]+x[4997]+x[4998]+x[4999]+x[5000]) +
        T(5)*x[4999]^3 +
        (x[5006]^2+x[4995]^2+x[4996]^2+x[4997]^2+x[4998]^2+x[4999]^2+x[5000]^2) + T(1)
    )^2)

    # ─── Constraints (K=5000..9998, ci=K-4999=1..4999) ───────────────────────────
    # Segment 1: ci=1..5, K=5000..5004 (K and all band members in identity window)
    ExaModels.@add_con(
        c,
        T(2) * x[i+4999] +
        (x[i+4994]+x[i+4995]+x[i+4996]+x[i+4997]+x[i+4998]+x[i+4999]+x[i+5000]) +
        T(5) * x[i+4999]^3 +
        (x[i+4994]^2+x[i+4995]^2+x[i+4996]^2+x[i+4997]^2+x[i+4998]^2+x[i+4999]^2+x[i+5000]^2) +
        T(1)
        for i in 1:5
    )

    # Segment 2: ci=6..11 (mixed identity/reversed band members — individual cases)
    # ci=6: K=5005, center=5005, band=[5000,5001,5002,5003,5004,5005,4994]
    ExaModels.@add_con(c, T(2)*x[5005]+(x[5000]+x[5001]+x[5002]+x[5003]+x[5004]+x[5005]+x[4994])+T(5)*x[5005]^3+(x[5000]^2+x[5001]^2+x[5002]^2+x[5003]^2+x[5004]^2+x[5005]^2+x[4994]^2)+T(1))
    # ci=7: K=5006, center=4994, band=[5001,5002,5003,5004,5005,4994,4993]
    ExaModels.@add_con(c, T(2)*x[4994]+(x[5001]+x[5002]+x[5003]+x[5004]+x[5005]+x[4994]+x[4993])+T(5)*x[4994]^3+(x[5001]^2+x[5002]^2+x[5003]^2+x[5004]^2+x[5005]^2+x[4994]^2+x[4993]^2)+T(1))
    # ci=8: K=5007, center=4993, band=[5002,5003,5004,5005,4994,4993,4992]
    ExaModels.@add_con(c, T(2)*x[4993]+(x[5002]+x[5003]+x[5004]+x[5005]+x[4994]+x[4993]+x[4992])+T(5)*x[4993]^3+(x[5002]^2+x[5003]^2+x[5004]^2+x[5005]^2+x[4994]^2+x[4993]^2+x[4992]^2)+T(1))
    # ci=9: K=5008, center=4992, band=[5003,5004,5005,4994,4993,4992,4991]
    ExaModels.@add_con(c, T(2)*x[4992]+(x[5003]+x[5004]+x[5005]+x[4994]+x[4993]+x[4992]+x[4991])+T(5)*x[4992]^3+(x[5003]^2+x[5004]^2+x[5005]^2+x[4994]^2+x[4993]^2+x[4992]^2+x[4991]^2)+T(1))
    # ci=10: K=5009, center=4991, band=[5004,5005,4994,4993,4992,4991,4990]
    ExaModels.@add_con(c, T(2)*x[4991]+(x[5004]+x[5005]+x[4994]+x[4993]+x[4992]+x[4991]+x[4990])+T(5)*x[4991]^3+(x[5004]^2+x[5005]^2+x[4994]^2+x[4993]^2+x[4992]^2+x[4991]^2+x[4990]^2)+T(1))
    # ci=11: K=5010, center=4990, band=[5005,4994,4993,4992,4991,4990,4989]
    ExaModels.@add_con(c, T(2)*x[4990]+(x[5005]+x[4994]+x[4993]+x[4992]+x[4991]+x[4990]+x[4989])+T(5)*x[4990]^3+(x[5005]^2+x[4994]^2+x[4993]^2+x[4992]^2+x[4991]^2+x[4990]^2+x[4989]^2)+T(1))

    # Segment 3: ci=12..N2m1 (fully reversed: K=ci+4999 > 5010, all positions reversed)
    # center=perm(K)=N-K=5001-ci; band=perm(K-5..K+1)={5000-ci..5006-ci}
    ExaModels.@add_con(
        c,
        T(2) * x[5001-i] +
        (x[5000-i]+x[5001-i]+x[5002-i]+x[5003-i]+x[5004-i]+x[5005-i]+x[5006-i]) +
        T(5) * x[5001-i]^3 +
        (x[5000-i]^2+x[5001-i]^2+x[5002-i]^2+x[5003-i]^2+x[5004-i]^2+x[5005-i]^2+x[5006-i]^2) +
        T(1)
        for i in 12:N2m1
    )

    return ExaModels.ExaModel(c; kwargs...)
end
