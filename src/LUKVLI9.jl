# LUKVLI9: Inequality version of LUKVLE9 (L-type constraints ≤ 0)
# Source: Problem 5.9, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), 6 L-type constraints
# All variables free; start x[i]=-1
function lukvli9_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert iseven(N) "N must be even"
    N2 = N ÷ 2

    perm = collect(1:N)
    if N >= 12
        perm[7] = N - 4
        perm[8] = 12
        perm[9] = N - 2
        perm[10] = N - 3
        perm[11] = N
        perm[12] = N - 1
        perm[N-4] = 11
        perm[N-3] = 10
        perm[N-2] = 9
        perm[N-1] = 8
        perm[N] = 7
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(-1))

    ExaModels.@add_obj(c, T(0.001) * x[2i-1]^2 for i in 1:3)
    ExaModels.@add_obj(c, -x[2i-1] + x[2i] for i in 1:3)
    ExaModels.@add_obj(c, exp(T(20) * (x[2i-1] - x[2i])) for i in 1:3)

    let a = perm[7], b = perm[8]
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    let a = perm[9], b = perm[10]
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    let a = perm[11], b = perm[12]
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    if N2 >= 10
        ExaModels.@add_obj(c, T(0.001) * x[2i-1]^2 for i in 7:(N2-3))
        ExaModels.@add_obj(c, -x[2i-1] + x[2i] for i in 7:(N2-3))
        ExaModels.@add_obj(c, exp(T(20) * (x[2i-1] - x[2i])) for i in 7:(N2-3))
    end

    let a = N - 5, b = perm[N-4]
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    let a = perm[N-3], b = perm[N-2]
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    let a = perm[N-1], b = perm[N]
        ExaModels.@add_obj(c, T(0.001) * x[a]^2)
        ExaModels.@add_obj(c, -x[a] + x[b])
        ExaModels.@add_obj(c, exp(T(20) * (x[a] - x[b])))
    end

    p1 = perm[1]; p2 = perm[2]; p3 = perm[3]; p4 = perm[4]; p5 = perm[5]; p6 = perm[6]
    pN = perm[N]; pNm1 = perm[N-1]; pNm2 = perm[N-2]; pNm3 = perm[N-3]
    pNm4 = perm[N-4]; pNm5 = perm[N-5]

    ExaModels.@add_con(
        c, T(4) * x[p1] + x[p2] + x[p3] - T(4) * x[p2]^2 - x[p3]^2 - x[p4]^2;
        lcon = T(-Inf), ucon = T(0)
    )
    ExaModels.@add_con(
        c,
        T(6) * x[p2] + x[p3] + x[p4] + T(8) * (x[p2]^3 - x[p2] * x[p1]) -
        T(4) * x[p3]^2 + x[p1]^2 - x[p4]^2 - x[p5]^2 - T(2);
        lcon = T(-Inf), ucon = T(0)
    )
    ExaModels.@add_con(
        c,
        T(6) * x[p3] + x[p4] + x[p5] - x[p1] + T(8) * (x[p3]^3 - x[p3] * x[p2]) -
        T(4) * x[p4]^2 + x[p2]^2 - x[p5]^2 + x[p1]^2 - x[p6]^2 - T(2);
        lcon = T(-Inf), ucon = T(0)
    )
    ExaModels.@add_con(
        c,
        T(6) * x[pNm2] + x[pNm1] + x[pN] - x[pNm4] - x[pNm5] +
        T(8) * (x[pNm2]^3 - x[pNm2] * x[pNm3]) - T(4) * x[pNm1]^2 + x[pNm3]^2 -
        x[pN]^2 + x[pNm4]^2 - T(2);
        lcon = T(-Inf), ucon = T(0)
    )
    ExaModels.@add_con(
        c,
        T(6) * x[pNm1] - x[pNm3] + x[pN] - x[pNm4] +
        T(8) * (x[pNm1]^3 - x[pNm1] * x[pNm2]) - T(4) * x[pN]^2 + x[pNm2]^2 +
        x[pNm3]^2 - T(2);
        lcon = T(-Inf), ucon = T(0)
    )
    ExaModels.@add_con(
        c,
        T(2) * x[pN] - x[pNm3] - x[pNm2] + T(8) * (x[pN]^3 - x[pN] * x[pNm1]) +
        x[pNm1]^2 + x[pNm2]^2;
        lcon = T(-Inf), ucon = T(0)
    )

    return ExaModels.ExaModel(c; kwargs...)
end
