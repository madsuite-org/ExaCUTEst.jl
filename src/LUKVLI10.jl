# LUKVLI10: Inequality version of LUKVLE10 (L-type constraints ≤ 0)
# Source: Problem 5.10, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), N-2=9998 L-type constraints
# All variables free; start x[odd]=-1, x[even]=1
function lukvli10_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert iseven(N) "N must be even"
    N2 = N ÷ 2

    perm = collect(1:N)
    if N >= 2
        perm[1], perm[N-1] = N - 1, 1
    end

    start = T[isodd(perm[c]) ? T(-1) : T(1) for c in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    let a1 = perm[1], b1 = perm[2]
        ExaModels.@add_obj(c, (x[a1]^2)^(x[b1]^2 + T(1)))
        ExaModels.@add_obj(c, (x[b1]^2)^(x[a1]^2 + T(1)))
    end
    if N2 >= 3
        ExaModels.@add_obj(c, (x[2i-1]^2)^(x[2i]^2 + T(1)) for i in 2:(N2-1))
        ExaModels.@add_obj(c, (x[2i]^2)^(x[2i-1]^2 + T(1)) for i in 2:(N2-1))
    end
    let aN = perm[N-1], bN = perm[N]
        ExaModels.@add_obj(c, (x[aN]^2)^(x[bN]^2 + T(1)))
        ExaModels.@add_obj(c, (x[bN]^2)^(x[aN]^2 + T(1)))
    end

    let p1 = perm[1]
        ExaModels.@add_con(
            c,
            T(3) * x[2] - x[p1] - T(2) * x[3] - T(2) * x[2]^2 + T(1);
            lcon = T(-Inf), ucon = T(0)
        )
    end

    if N >= 7
        ExaModels.@add_con(
            c,
            T(3) * x[k+1] - x[k] - T(2) * x[k+2] - T(2) * x[k+1]^2 + T(1) for
            k in 2:(N-4);
            lcon = T(-Inf), ucon = T(0)
        )
    end

    let pNm1 = perm[N-1]
        ExaModels.@add_con(
            c,
            T(3) * x[N-2] - x[N-3] - T(2) * x[pNm1] - T(2) * x[N-2]^2 + T(1);
            lcon = T(-Inf), ucon = T(0)
        )
        ExaModels.@add_con(
            c,
            T(3) * x[pNm1] - x[N-2] - T(2) * x[N] - T(2) * x[pNm1]^2 + T(1);
            lcon = T(-Inf), ucon = T(0)
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end
