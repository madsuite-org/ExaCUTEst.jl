# LUKVLI2: Inequality version of LUKVLE2 (L-type constraints ≤ 0)
# Source: Problem 5.2, Luksan & Vlcek (1999), TR 767
# N=10000 variables, N/2-1=4999 L-type inequality constraints
# All variables free; start: odd→-2, even→1
#
# Objective: same Chained Wood + Broyden banded structure as LUKVLE2.
# Constraints: same Broyden banded formula with lcon=-Inf, ucon=0.
function lukvli2_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert N == 10000 "LUKVLI2 uses a fixed sifdecoder variable ordering only valid for N=10000"
    N2m1 = N ÷ 2 - 1  # = 4999

    start = T[isodd(i) ? T(-2) : T(1) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # ─── Objective (identical to LUKVLE2) ────────────────────────────────────────
    ExaModels.@add_obj(c, T(100) * (-x[10000-2i] + x[10001-2i]^2)^2 for i in 1:2496)
    ExaModels.@add_obj(c, (x[10001-2i] - T(1))^2 for i in 1:2496)
    ExaModels.@add_obj(c, (-x[9999-2i] - T(1))^2 for i in 1:2496)
    ExaModels.@add_obj(c, T(10) * (x[10000-2i] + x[9998-2i] - T(2))^2 for i in 1:2496)
    ExaModels.@add_obj(c, (x[10000-2i] - x[10001-2i])^2 / T(10) for i in 1:2496)

    ExaModels.@add_obj(c, T(100) * (-x[10000-2i] + x[10001-2i]^2)^2 for i in 2504:4999)
    ExaModels.@add_obj(c, (x[10001-2i] - T(1))^2 for i in 2504:4999)
    ExaModels.@add_obj(c, (-x[9999-2i] - T(1))^2 for i in 2504:4999)
    ExaModels.@add_obj(c, T(10) * (x[10000-2i] + x[9998-2i] - T(2))^2 for i in 2504:4998)
    ExaModels.@add_obj(c, (x[10000-2i] - x[10001-2i])^2 / T(10) for i in 2504:4999)

    ExaModels.@add_obj(c, T(10) * (x[2] + x[10000] - T(2))^2)

    ExaModels.@add_obj(c, T(100) * (-x[5006] + x[5007]^2)^2)
    ExaModels.@add_obj(c, (x[5007] - T(1))^2)
    ExaModels.@add_obj(c, (-x[4995] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[5006] + x[4996] - T(2))^2)
    ExaModels.@add_obj(c, (x[5006] - x[5007])^2 / T(10))

    ExaModels.@add_obj(c, T(100) * (-x[4996] + x[4995]^2)^2)
    ExaModels.@add_obj(c, (x[4995] - T(1))^2)
    ExaModels.@add_obj(c, (-x[4997] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[4996] + x[4998] - T(2))^2)
    ExaModels.@add_obj(c, (x[4996] - x[4995])^2 / T(10))

    ExaModels.@add_obj(c, T(100) * (-x[4998] + x[4997]^2)^2)
    ExaModels.@add_obj(c, (x[4997] - T(1))^2)
    ExaModels.@add_obj(c, (-x[4999] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[4998] + x[5000] - T(2))^2)
    ExaModels.@add_obj(c, (x[4998] - x[4997])^2 / T(10))

    ExaModels.@add_obj(c, T(100) * (-x[5000] + x[4999]^2)^2)
    ExaModels.@add_obj(c, (x[4999] - T(1))^2)
    ExaModels.@add_obj(c, (-x[5001] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[5000] + x[5002] - T(2))^2)
    ExaModels.@add_obj(c, (x[5000] - x[4999])^2 / T(10))

    ExaModels.@add_obj(c, T(100) * (-x[5002] + x[5001]^2)^2)
    ExaModels.@add_obj(c, (x[5001] - T(1))^2)
    ExaModels.@add_obj(c, (-x[5003] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[5002] + x[5004] - T(2))^2)
    ExaModels.@add_obj(c, (x[5002] - x[5001])^2 / T(10))

    ExaModels.@add_obj(c, T(100) * (-x[5004] + x[5003]^2)^2)
    ExaModels.@add_obj(c, (x[5003] - T(1))^2)
    ExaModels.@add_obj(c, (-x[5005] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[5004] + x[4994] - T(2))^2)
    ExaModels.@add_obj(c, (x[5004] - x[5003])^2 / T(10))

    ExaModels.@add_obj(c, T(100) * (-x[4994] + x[5005]^2)^2)
    ExaModels.@add_obj(c, (x[5005] - T(1))^2)
    ExaModels.@add_obj(c, (-x[4993] - T(1))^2)
    ExaModels.@add_obj(c, T(10) * (x[4994] + x[4992] - T(2))^2)
    ExaModels.@add_obj(c, (x[4994] - x[5005])^2 / T(10))

    ExaModels.@add_obj(c, T(90) * (-x[9998-2i] + x[9999-2i]^2)^2 for i in 1:5)

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
    # C(I=2497..2502): wood positions fall in identity window
    ExaModels.@add_obj(c, T(90)*(-x[4996]+x[4995]^2+T(2)*x[7503]+(x[7508]+x[7507]+x[7506]+x[7505]+x[7504]+x[7503]+x[7502])+T(5)*x[7503]^3+(x[7508]^2+x[7507]^2+x[7506]^2+x[7505]^2+x[7504]^2+x[7503]^2+x[7502]^2)+T(1))^2)
    ExaModels.@add_obj(c, T(90)*(-x[4998]+x[4997]^2+T(2)*x[7502]+(x[7507]+x[7506]+x[7505]+x[7504]+x[7503]+x[7502]+x[7501])+T(5)*x[7502]^3+(x[7507]^2+x[7506]^2+x[7505]^2+x[7504]^2+x[7503]^2+x[7502]^2+x[7501]^2)+T(1))^2)
    ExaModels.@add_obj(c, T(90)*(-x[5000]+x[4999]^2+T(2)*x[7501]+(x[7506]+x[7505]+x[7504]+x[7503]+x[7502]+x[7501]+x[7500])+T(5)*x[7501]^3+(x[7506]^2+x[7505]^2+x[7504]^2+x[7503]^2+x[7502]^2+x[7501]^2+x[7500]^2)+T(1))^2)
    ExaModels.@add_obj(c, T(90)*(-x[5002]+x[5001]^2+T(2)*x[7500]+(x[7505]+x[7504]+x[7503]+x[7502]+x[7501]+x[7500]+x[7499])+T(5)*x[7500]^3+(x[7505]^2+x[7504]^2+x[7503]^2+x[7502]^2+x[7501]^2+x[7500]^2+x[7499]^2)+T(1))^2)
    ExaModels.@add_obj(c, T(90)*(-x[5004]+x[5003]^2+T(2)*x[7499]+(x[7504]+x[7503]+x[7502]+x[7501]+x[7500]+x[7499]+x[7498])+T(5)*x[7499]^3+(x[7504]^2+x[7503]^2+x[7502]^2+x[7501]^2+x[7500]^2+x[7499]^2+x[7498]^2)+T(1))^2)
    ExaModels.@add_obj(c, T(90)*(-x[4994]+x[5005]^2+T(2)*x[7498]+(x[7503]+x[7502]+x[7501]+x[7500]+x[7499]+x[7498]+x[7497])+T(5)*x[7498]^3+(x[7503]^2+x[7502]^2+x[7501]^2+x[7500]^2+x[7499]^2+x[7498]^2+x[7497]^2)+T(1))^2)

    ExaModels.@add_obj(c, T(90) * (
        -x[10] + x[11]^2 + T(2)*x[5006] +
        (x[5011]+x[5010]+x[5009]+x[5008]+x[5007]+x[5006]+x[4995]) +
        T(5)*x[5006]^3 +
        (x[5011]^2+x[5010]^2+x[5009]^2+x[5008]^2+x[5007]^2+x[5006]^2+x[4995]^2) + T(1)
    )^2)
    ExaModels.@add_obj(c, T(90) * (
        -x[8] + x[9]^2 + T(2)*x[4995] +
        (x[5010]+x[5009]+x[5008]+x[5007]+x[5006]+x[4995]+x[4996]) +
        T(5)*x[4995]^3 +
        (x[5010]^2+x[5009]^2+x[5008]^2+x[5007]^2+x[5006]^2+x[4995]^2+x[4996]^2) + T(1)
    )^2)
    ExaModels.@add_obj(c, T(90) * (
        -x[6] + x[7]^2 + T(2)*x[4996] +
        (x[5009]+x[5008]+x[5007]+x[5006]+x[4995]+x[4996]+x[4997]) +
        T(5)*x[4996]^3 +
        (x[5009]^2+x[5008]^2+x[5007]^2+x[5006]^2+x[4995]^2+x[4996]^2+x[4997]^2) + T(1)
    )^2)
    ExaModels.@add_obj(c, T(90) * (
        -x[4] + x[5]^2 + T(2)*x[4997] +
        (x[5008]+x[5007]+x[5006]+x[4995]+x[4996]+x[4997]+x[4998]) +
        T(5)*x[4997]^3 +
        (x[5008]^2+x[5007]^2+x[5006]^2+x[4995]^2+x[4996]^2+x[4997]^2+x[4998]^2) + T(1)
    )^2)
    ExaModels.@add_obj(c, T(90) * (
        -x[2] + x[3]^2 + T(2)*x[4998] +
        (x[5007]+x[5006]+x[4995]+x[4996]+x[4997]+x[4998]+x[4999]) +
        T(5)*x[4998]^3 +
        (x[5007]^2+x[5006]^2+x[4995]^2+x[4996]^2+x[4997]^2+x[4998]^2+x[4999]^2) + T(1)
    )^2)
    ExaModels.@add_obj(c, T(90) * (
        -x[10000] + x[1]^2 + T(2)*x[4999] +
        (x[5006]+x[4995]+x[4996]+x[4997]+x[4998]+x[4999]+x[5000]) +
        T(5)*x[4999]^3 +
        (x[5006]^2+x[4995]^2+x[4996]^2+x[4997]^2+x[4998]^2+x[4999]^2+x[5000]^2) + T(1)
    )^2)

    # ─── Constraints (L-type ≤ 0) ────────────────────────────────────────────────
    # Segment 1: ci=1..5
    ExaModels.@add_con(
        c,
        T(2) * x[i+4999] +
        (x[i+4994]+x[i+4995]+x[i+4996]+x[i+4997]+x[i+4998]+x[i+4999]+x[i+5000]) +
        T(5) * x[i+4999]^3 +
        (x[i+4994]^2+x[i+4995]^2+x[i+4996]^2+x[i+4997]^2+x[i+4998]^2+x[i+4999]^2+x[i+5000]^2) +
        T(1)
        for i in 1:5;
        lcon = T(-Inf), ucon = T(0)
    )
    # Segment 2: ci=6..11
    ExaModels.@add_con(c, T(2)*x[5005]+(x[5000]+x[5001]+x[5002]+x[5003]+x[5004]+x[5005]+x[4994])+T(5)*x[5005]^3+(x[5000]^2+x[5001]^2+x[5002]^2+x[5003]^2+x[5004]^2+x[5005]^2+x[4994]^2)+T(1); lcon=T(-Inf), ucon=T(0))
    ExaModels.@add_con(c, T(2)*x[4994]+(x[5001]+x[5002]+x[5003]+x[5004]+x[5005]+x[4994]+x[4993])+T(5)*x[4994]^3+(x[5001]^2+x[5002]^2+x[5003]^2+x[5004]^2+x[5005]^2+x[4994]^2+x[4993]^2)+T(1); lcon=T(-Inf), ucon=T(0))
    ExaModels.@add_con(c, T(2)*x[4993]+(x[5002]+x[5003]+x[5004]+x[5005]+x[4994]+x[4993]+x[4992])+T(5)*x[4993]^3+(x[5002]^2+x[5003]^2+x[5004]^2+x[5005]^2+x[4994]^2+x[4993]^2+x[4992]^2)+T(1); lcon=T(-Inf), ucon=T(0))
    ExaModels.@add_con(c, T(2)*x[4992]+(x[5003]+x[5004]+x[5005]+x[4994]+x[4993]+x[4992]+x[4991])+T(5)*x[4992]^3+(x[5003]^2+x[5004]^2+x[5005]^2+x[4994]^2+x[4993]^2+x[4992]^2+x[4991]^2)+T(1); lcon=T(-Inf), ucon=T(0))
    ExaModels.@add_con(c, T(2)*x[4991]+(x[5004]+x[5005]+x[4994]+x[4993]+x[4992]+x[4991]+x[4990])+T(5)*x[4991]^3+(x[5004]^2+x[5005]^2+x[4994]^2+x[4993]^2+x[4992]^2+x[4991]^2+x[4990]^2)+T(1); lcon=T(-Inf), ucon=T(0))
    ExaModels.@add_con(c, T(2)*x[4990]+(x[5005]+x[4994]+x[4993]+x[4992]+x[4991]+x[4990]+x[4989])+T(5)*x[4990]^3+(x[5005]^2+x[4994]^2+x[4993]^2+x[4992]^2+x[4991]^2+x[4990]^2+x[4989]^2)+T(1); lcon=T(-Inf), ucon=T(0))
    # Segment 3: ci=12..N2m1
    ExaModels.@add_con(
        c,
        T(2) * x[5001-i] +
        (x[5000-i]+x[5001-i]+x[5002-i]+x[5003-i]+x[5004-i]+x[5005-i]+x[5006-i]) +
        T(5) * x[5001-i]^3 +
        (x[5000-i]^2+x[5001-i]^2+x[5002-i]^2+x[5003-i]^2+x[5004-i]^2+x[5005-i]^2+x[5006-i]^2) +
        T(1)
        for i in 12:N2m1;
        lcon = T(-Inf), ucon = T(0)
    )

    return ExaModels.ExaModel(c; kwargs...)
end
