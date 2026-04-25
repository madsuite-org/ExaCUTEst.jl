# LUKVLI12: Inequality version of LUKVLE12 (L-type constraints ≤ 0)
# Source: Problem 5.12, Luksan & Vlcek (1999), TR 767
# N=9997 variables (default), NC=7497 L-type constraints
function lukvli12_model(N = 9997; T = Float64, backend = nothing, kwargs...)
    @assert N % 4 == 1 "N must satisfy N ≡ 1 (mod 4) for LUKVLI12"
    NB = (N - 1) ÷ 4
    NC = 3 * NB

    start = T[
        mod(i, 4) == 1 ? T(2) :
        mod(i, 4) == 2 ? T(1.5) :
        mod(i, 4) == 3 ? T(-1) : T(0.5) for i in 1:N
    ]
    start[NC+1] = T(-1)
    start[NC+2] = T(1.5)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    I_exc = (NC + 3) ÷ 4

    ExaModels.@add_obj(c, (x[4*I_exc-3] - x[NC+2])^2)
    ExaModels.@add_obj(c, (x[4i-3] - x[4i-2])^2 for i in 1:I_exc-1)
    ExaModels.@add_obj(c, (x[4i-3] - x[4i-2])^2 for i in I_exc+1:NB)

    ExaModels.@add_obj(c, (x[4i-2] - x[4i-1])^2 for i in 1:NB)

    ExaModels.@add_obj(c, (x[NC+1] - x[4*I_exc])^4)
    ExaModels.@add_obj(c, (x[4i-1] - x[4i])^4 for i in 1:I_exc-1)
    ExaModels.@add_obj(c, (x[4i-1] - x[4i])^4 for i in I_exc+1:NB)

    ExaModels.@add_obj(c, (x[4i] - x[4i+1])^4 for i in 1:NB)

    ExaModels.@add_con(c, x[1] + x[2]^2 + x[3]^2 - T(3); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, x[2] + x[4] + x[3]^2 - T(1); lcon = T(-Inf), ucon = T(0))

    q1_data = T[((i + 2) % 3 == 1) ? T(1) : T(0) for i in 1:NC-2]
    q2_data = T[((i + 2) % 3 == 2) ? T(1) : T(0) for i in 1:NC-2]
    q3_data = T[((i + 2) % 3 == 0) ? T(1) : T(0) for i in 1:NC-2]
    qe_data = T[(i == NC - 3) ? T(1) : T(0) for i in 1:NC-2]

    ExaModels.@add_par(c, q1, q1_data)
    ExaModels.@add_par(c, q2, q2_data)
    ExaModels.@add_par(c, q3, q3_data)
    ExaModels.@add_par(c, qe, qe_data)

    ExaModels.@add_con(c,
        q1[i] * (x[i+2] + x[i+3]^2 + x[i+4]^2 - T(3)) +
        q2[i] * (x[i+2] + x[i+4] + qe[i] * (x[NC+2] - x[i+4]) + x[i+3]^2 - T(1)) +
        q3[i] * (x[i] * x[NC+1] - T(1))
        for i in 1:NC-2;
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
