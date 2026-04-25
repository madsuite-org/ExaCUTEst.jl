# LUKVLI15: Inequality version of LUKVLE15 (L-type constraints ≤ 0)
# Source: Problem 5.15, Luksan & Vlcek (1999), TR 767
# N=9997 variables (default), NC=7497 L-type constraints
function lukvli15_model(N = 9997; T = Float64, backend = nothing, kwargs...)
    @assert N % 4 == 1 "N must satisfy N ≡ 1 (mod 4) for LUKVLI15"
    NB = (N - 1) ÷ 4
    NC = 3 * NB

    start = T[
        mod(i, 4) == 1 ? T(35) :
        mod(i, 4) == 2 ? T(11) :
        mod(i, 4) == 3 ? T(5) : T(-5) for i in 1:N
    ]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    ExaModels.@add_obj(c, (x[4i-3] - x[4i-2])^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i-2] - x[4i-1])^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i-1] - x[4i])^4 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i] - x[4i+1])^4 for i in 1:NB)

    ExaModels.@add_con(c, x[j]^2 + T(2) * x[j+1] + T(3) * x[j+2] - T(6) for j in 1:NC; lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
