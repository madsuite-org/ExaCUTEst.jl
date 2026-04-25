# LUKVLI13: Inequality version of LUKVLE13 (L-type constraints ≤ 0)
# Source: Problem 5.13, Luksan & Vlcek (1999), TR 767
# N=9998 variables (default), NC=6664 L-type constraints
function lukvli13_model(N = 9998; T = Float64, backend = nothing, kwargs...)
    @assert N % 3 == 2 "N must satisfy N ≡ 2 (mod 3) for LUKVLI13"
    NB = (N - 2) ÷ 3
    NC = 2 * NB

    start = T[mod(i, 3) == 1 ? T(3) : mod(i, 3) == 2 ? T(5) : T(-3) for i in 1:N]
    start[1]    = T(5)
    start[NC+1] = T(3)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    I_obj2_exc = (NC + 2) ÷ 3
    I_obj3_exc = (NC - 1) ÷ 3

    ExaModels.@add_obj(c, (x[NC+1] - T(1))^2)
    ExaModels.@add_obj(c, (x[3i-2] - T(1))^2 for i in 2:NB)

    ExaModels.@add_obj(c, (x[1] - x[3*I_obj2_exc])^2)
    ExaModels.@add_obj(c, (x[3i-1] - x[3i])^2 for i in 1:I_obj2_exc-1)
    ExaModels.@add_obj(c, (x[3i-1] - x[3i])^2 for i in I_obj2_exc+1:NB)

    ExaModels.@add_obj(c, (x[3*I_obj3_exc+1] - x[1])^4)
    ExaModels.@add_obj(c, (x[3i+1] - x[3i+2])^4 for i in 1:I_obj3_exc-1)
    ExaModels.@add_obj(c, (x[3i+1] - x[3i+2])^4 for i in I_obj3_exc+1:NB)

    ExaModels.@add_con(c, x[NC+1] + x[2]^2 + x[3] + x[4] + T(4) * x[5] - T(5); lcon = T(-Inf), ucon = T(0))

    podd_m_data  = T[((i + 1) % 2 == 1) ? T(1) : T(0) for i in 1:NC-5]
    peven_m_data = T[((i + 1) % 2 == 0) ? T(1) : T(0) for i in 1:NC-5]

    ExaModels.@add_par(c, podd_m,  podd_m_data)
    ExaModels.@add_par(c, peven_m, peven_m_data)

    ExaModels.@add_con(c,
        podd_m[i]  * (x[i+1] + x[i+2]^2 + x[i+3] + x[i+4] + T(4) * x[i+5] - T(5)) +
        peven_m[i] * (-T(2) * x[i+3] - T(2) * x[i+4] + x[i+2]^2 - T(3))
        for i in 1:NC-5;
        lcon = T(-Inf), ucon = T(0))

    ExaModels.@add_con(c, x[NC-3] + x[NC-2]^2 + x[NC-1] + x[NC] + T(4) * x[1] - T(5); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, -T(2) * x[NC] - T(2) * x[1] + x[NC-1]^2 - T(3); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, x[NC-1] + x[NC]^2 + x[1] + x[NC+2] + T(4) * x[NC+3] - T(5); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, -T(2) * x[NC+2] - T(2) * x[NC+3] + x[1]^2 - T(3); lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
