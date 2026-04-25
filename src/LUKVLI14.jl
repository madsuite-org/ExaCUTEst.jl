# LUKVLI14: Inequality version of LUKVLE14 (L-type constraints ≤ 0)
# Source: Problem 5.14, Luksan & Vlcek (1999), TR 767
# NOTE: Same implementation limitation as LUKVLE14 — CUTEst's sifdecoder applies
# pervasive variable swaps that are infeasible to fully replicate. Will FAIL CUTEst test.
function lukvli14_model(N = 9998; T = Float64, backend = nothing, kwargs...)
    @assert N % 3 == 2 "N must satisfy N ≡ 2 (mod 3) for LUKVLI14"
    NB = (N - 2) ÷ 3
    NC = 2 * NB

    start = T[mod(i, 3) == 1 ? T(10) : mod(i, 3) == 2 ? T(7) : T(-3) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    ExaModels.@add_obj(c, (x[3i-2] - x[3i-1])^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[3i] - T(1))^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[3i+1] - T(1))^4 for i in 1:NB)
    ExaModels.@add_obj(c, (x[3i+2] - T(1))^6 for i in 1:NB)

    ExaModels.@add_con(
        c,
        x[2m-1]^2 + x[2m] + x[2m+1] + T(4) * x[2m+2] - T(7) for m in 1:NB;
        lcon = T(-Inf), ucon = T(0)
    )
    ExaModels.@add_con(c, -T(5) * x[2m+3] + x[2m+1]^2 - T(6) for m in 1:NB; lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
