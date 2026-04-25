# LUKVLI17: Inequality version of LUKVLE17 (G-type constraints ≥ 0)
# Source: Problem 5.17, Luksan & Vlcek (1999), TR 767
# N=9997 variables (default), NC=7497 G-type constraints
function lukvli17_model(N = 9997; T = Float64, backend = nothing, kwargs...)
    @assert N % 4 == 1 "N must satisfy N ≡ 1 (mod 4) for LUKVLI17"
    NB = (N - 1) ÷ 4
    NC = 3 * NB

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(2))

    ExaModels.@add_obj(c, (T(4) * x[4i-3] - x[4i-2])^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i-2] + x[4i-1] - T(2))^4 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i] - T(1))^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i+1] - T(1))^2 for i in 1:NB)

    ExaModels.@add_con(c, x[1]^2 + T(3) * x[2]; lcon = T(0), ucon = T(Inf))

    ps1_data = T[((i + 1) % 3 == 1) ? T(1) : T(0) for i in 1:NC-1]
    ps2_data = T[((i + 1) % 3 == 2) ? T(1) : T(0) for i in 1:NC-1]
    ps3_data = T[((i + 1) % 3 == 0) ? T(1) : T(0) for i in 1:NC-1]

    ExaModels.@add_par(c, ps1, ps1_data)
    ExaModels.@add_par(c, ps2, ps2_data)
    ExaModels.@add_par(c, ps3, ps3_data)

    ExaModels.@add_con(c,
        ps1[i] * (x[i+1]^2 + T(3) * x[i+2]) +
        ps2[i] * (x[i+2]^2 + x[i+3] - T(2) * x[i+4]) +
        ps3[i] * (x[i]^2 - x[i+3])
        for i in 1:NC-1;
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
