# LUKVLI7: Inequality version of LUKVLE7 (L-type constraints ≤ 0)
# Source: Problem 5.7, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), 4 L-type constraints
# All variables free; start x[i]=1
function lukvli7_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    perm = collect(1:N)
    if N >= 7
        perm[5], perm[N]   = N,   5
        perm[6], perm[N-1] = N-1, 6
        perm[7], perm[N-2] = N-2, 7
    end
    inv_perm = invperm(perm)

    start = ones(T, N)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    k_par = T[T(inv_perm[c]) for c in 1:N]
    ExaModels.@add_par(c, kpar, k_par)

    si_coeff = T[inv_perm[c] < N ? T(2) : T(-(N - 1)) for c in 1:N]
    ExaModels.@add_par(c, sipar, si_coeff)

    half_Np1 = T(N + 1) / T(2)
    ExaModels.@add_obj(c, -kpar[i] * cos(x[i]) + sipar[i] * sin(x[i]) + half_Np1 for i in 1:N)

    p_xNm2 = perm[N-2]
    p_xNm1 = perm[N-1]
    p_xN   = perm[N]
    p_xNm3 = perm[N-3]

    ExaModels.@add_con(c, T(4) * x[1] + x[2] - T(4) * x[2]^2 - x[3]^2; lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(
        c,
        T(6) * x[2] + x[3] + T(8) * (x[2]^3 - x[2] * x[1]) - T(4) * x[3]^2 - x[4]^2 - T(2);
        lcon = T(-Inf), ucon = T(0)
    )
    ExaModels.@add_con(
        c,
        T(6) * x[p_xNm1] - x[p_xNm3] + T(8) * (x[p_xNm1]^3 - x[p_xNm1] * x[p_xNm2]) -
        T(4) * x[p_xN]^2 + x[p_xNm2]^2 - T(2);
        lcon = T(-Inf), ucon = T(0)
    )
    ExaModels.@add_con(
        c,
        T(2) * x[p_xN] - x[p_xNm2] + T(8) * (x[p_xN]^3 - x[p_xN] * x[p_xNm1]) +
        x[p_xNm1]^2;
        lcon = T(-Inf), ucon = T(0)
    )

    return ExaModels.ExaModel(c; kwargs...)
end
