# LUKVLI8: Inequality version of LUKVLE8 (L-type constraints ≤ 0)
# Source: Problem 5.8, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), N-2=9998 L-type constraints
# All variables free; start x[odd]=-1, x[even]=2
function lukvli8_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert N % 5 == 0 "N must be divisible by 5"
    N5 = N ÷ 5
    N2 = N - 2

    lambda1 = T(-0.002008)
    lambda2 = T(-0.001900)
    lambda3 = T(-0.000261)
    lamba1p10 = lambda1 + T(10)
    lamba3m1 = lambda3 - T(1)

    h = T(1) / T(N + 1)
    h2h2 = T(0.5) * h^2

    perm = collect(1:N)
    if N >= 2
        perm[1], perm[N-1] = N - 1, 1
    end
    p = perm

    start = T[isodd(perm[c]) ? T(-1) : T(2) for c in 1:N]

    P = T[(k + 1) * h + T(1) for k in 1:N2]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    ExaModels.@add_par(c, P, P)

    if N5 >= 3
        ExaModels.@add_obj(
            c,
            exp(x[5i] * x[5i-1] * x[5i-2] * x[5i-3] * x[5i-4]) for i in 2:(N5-1)
        )
        ExaModels.@add_obj(
            c,
            T(10) *
            (x[5i]^2 + x[5i-1]^2 + x[5i-2]^2 + x[5i-3]^2 + x[5i-4]^2 - lamba1p10)^2
            for i in 2:(N5-1)
        )
        ExaModels.@add_obj(
            c,
            T(10) * (x[5i-3] * x[5i-2] - T(5) * x[5i-1] * x[5i] - lambda2)^2
            for i in 2:(N5-1)
        )
        ExaModels.@add_obj(
            c,
            T(10) * (x[5i-4]^3 + x[5i-3]^3 - lamba3m1)^2 for i in 2:(N5-1)
        )
    end

    let p1 = p[1]
        ExaModels.@add_obj(c, exp(x[5] * x[4] * x[3] * x[2] * x[p1]))
        ExaModels.@add_obj(
            c,
            T(10) * (x[5]^2 + x[4]^2 + x[3]^2 + x[2]^2 + x[p1]^2 - lamba1p10)^2,
        )
        ExaModels.@add_obj(c, T(10) * (x[2] * x[3] - T(5) * x[4] * x[5] - lambda2)^2)
        ExaModels.@add_obj(c, T(10) * (x[p1]^3 + x[2]^3 - lamba3m1)^2)
    end

    let pNm1 = p[N-1]
        ExaModels.@add_obj(c, exp(x[N] * x[pNm1] * x[N-2] * x[N-3] * x[N-4]))
        ExaModels.@add_obj(
            c,
            T(10) *
            (x[N]^2 + x[pNm1]^2 + x[N-2]^2 + x[N-3]^2 + x[N-4]^2 - lamba1p10)^2,
        )
        ExaModels.@add_obj(
            c,
            T(10) * (x[N-3] * x[N-2] - T(5) * x[pNm1] * x[N] - lambda2)^2,
        )
        ExaModels.@add_obj(c, T(10) * (x[N-4]^3 + x[N-3]^3 - lamba3m1)^2)
    end

    let p1 = p[1]
        ExaModels.@add_con(
            c,
            T(2) * x[2] - x[p1] - x[3] + h2h2 * (x[2] + P[1])^2;
            lcon = T(-Inf), ucon = T(0)
        )
    end

    if N >= 7
        ExaModels.@add_con(
            c,
            T(2) * x[k+1] - x[k] - x[k+2] + h2h2 * (x[k+1] + P[k])^2 for
            k in 2:(N-4);
            lcon = T(-Inf), ucon = T(0)
        )
    end

    let pNm1 = p[N-1]
        ExaModels.@add_con(
            c,
            T(2) * x[N-2] - x[N-3] - x[pNm1] + h2h2 * (x[N-2] + P[N-3])^2;
            lcon = T(-Inf), ucon = T(0)
        )
        ExaModels.@add_con(
            c,
            T(2) * x[pNm1] - x[N-2] - x[N] + h2h2 * (x[pNm1] + P[N-2])^2;
            lcon = T(-Inf), ucon = T(0)
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end
