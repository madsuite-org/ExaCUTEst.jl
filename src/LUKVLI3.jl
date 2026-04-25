# LUKVLI3: Inequality version of LUKVLE3 (G-type C1, L-type C2)
# Source: Problem 5.3, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), 2 inequality constraints
# All variables free; start: X(4k+1)=3, X(4k+2)=-1, X(4k+3)=0, X(4k+4)=1
function lukvli3_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert iseven(N) "N must be even"
    @assert N >= 6 "N must be >= 6"
    N2m1 = N ÷ 2 - 1

    perm = collect(1:N)
    perm[3] = N
    perm[4] = N - 1
    perm[N-1] = 4
    perm[N] = 3

    sif_start = T[if k % 4 == 1; T(3) elseif k % 4 == 2; T(-1) elseif k % 4 == 3; T(0) else T(1) end for k in 1:N]
    start = T[sif_start[invperm(perm)[c]] for c in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Group I=1: SIF pos (1,2,3,4) → CUTEst (1, 2, perm[3]=N, perm[4]=N-1)
    let pa = perm[1], pb = perm[2], pc = perm[3], pd = perm[4]
        ExaModels.@add_obj(c, (x[pa] + T(10) * x[pb])^2)
        ExaModels.@add_obj(c, T(5) * (x[pc] - x[pd])^2)
        ExaModels.@add_obj(c, (x[pb] - T(2) * x[pc])^4)
        ExaModels.@add_obj(c, T(10) * (x[pa] - x[pd])^4)
    end

    # Group I=2: SIF pos (3,4,5,6) → CUTEst (perm[3]=N, perm[4]=N-1, perm[5], perm[6])
    let pa = perm[3], pb = perm[4], pc = perm[5], pd = perm[6]
        ExaModels.@add_obj(c, (x[pa] + T(10) * x[pb])^2)
        ExaModels.@add_obj(c, T(5) * (x[pc] - x[pd])^2)
        ExaModels.@add_obj(c, (x[pb] - T(2) * x[pc])^4)
        ExaModels.@add_obj(c, T(10) * (x[pa] - x[pd])^4)
    end

    # Interior groups I=3..N/2-2: all natural positions (identity permutation)
    if N2m1 >= 4
        ExaModels.@add_obj(c, (x[2i-1] + T(10) * x[2i])^2 for i in 3:(N2m1-1))
        ExaModels.@add_obj(c, T(5) * (x[2i+1] - x[2i+2])^2 for i in 3:(N2m1-1))
        ExaModels.@add_obj(c, (x[2i] - T(2) * x[2i+1])^4 for i in 3:(N2m1-1))
        ExaModels.@add_obj(c, T(10) * (x[2i-1] - x[2i+2])^4 for i in 3:(N2m1-1))
    end

    # Group I=N/2-1: SIF pos (N-3,N-2,N-1,N) → CUTEst (N-3, N-2, perm[N-1]=4, perm[N]=3)
    if N2m1 > 2
        let pa = perm[N-3], pb = perm[N-2], pc = perm[N-1], pd = perm[N]
            ExaModels.@add_obj(c, (x[pa] + T(10) * x[pb])^2)
            ExaModels.@add_obj(c, T(5) * (x[pc] - x[pd])^2)
            ExaModels.@add_obj(c, (x[pb] - T(2) * x[pc])^4)
            ExaModels.@add_obj(c, T(10) * (x[pa] - x[pd])^4)
        end
    end

    # C1 (XG): G-type ≥ 0
    ExaModels.@add_con(
        c,
        T(3) * x[1]^3 + sin(x[1] - x[2]) * sin(x[1] + x[2]) + T(2) * x[2] - T(5);
        lcon = T(0), ucon = T(Inf)
    )

    # C2 (XL): L-type ≤ 0
    let pNm1 = perm[N-1], pN = perm[N]
        ExaModels.@add_con(
            c,
            T(4) * x[pNm1] - x[pNm1] * exp(x[pNm1] - x[pN]) - T(3);
            lcon = T(-Inf), ucon = T(0)
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end
