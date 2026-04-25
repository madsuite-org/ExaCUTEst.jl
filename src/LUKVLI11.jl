# LUKVLI11: Inequality version of LUKVLE11 (L-type constraints ≤ 0)
# Source: Problem 5.11, Luksan & Vlcek (1999), TR 767
# N=9998 variables (default), NC=6664 L-type constraints
function lukvli11_model(N = 9998; T = Float64, backend = nothing, kwargs...)
    @assert N % 3 == 2 "N must satisfy N ≡ 2 (mod 3) for LUKVLI11"
    NB = (N - 2) ÷ 3
    NC = 2 * NB

    start = T[mod(i, 3) == 1 ? T(2) : mod(i, 3) == 2 ? T(1.5) : T(0.5) for i in 1:N]
    start[2]    = T(2)
    start[NC+3] = T(1.5)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    I_obj1b = (NC + 5) ÷ 3
    I_obj3  = (NC + 2) ÷ 3

    ExaModels.@add_obj(c, (x[1] - x[NC+3])^2)
    ExaModels.@add_obj(c, (x[2] - x[3*I_obj1b-1])^2)
    ExaModels.@add_obj(c, (x[3i-2] - x[3i-1])^2 for i in 2:I_obj1b-1)
    ExaModels.@add_obj(c, (x[3i-2] - x[3i-1])^2 for i in I_obj1b+1:NB)

    ExaModels.@add_obj(c, (x[3i] - T(1))^2 for i in 1:NB)

    ExaModels.@add_obj(c, (x[2] - T(1))^4)
    ExaModels.@add_obj(c, (x[3i+1] - T(1))^4 for i in 1:I_obj3-1)
    ExaModels.@add_obj(c, (x[3i+1] - T(1))^4 for i in I_obj3+1:NB)

    ExaModels.@add_obj(c, (x[3i+2] - T(1))^6 for i in 1:NB)

    podd_data  = T[j % 2 == 1 ? T(1) : T(0) for j in 1:NC]
    peven_data = T[j % 2 == 0 ? T(1) : T(0) for j in 1:NC]
    peNC1_data = T[j == NC - 1 ? T(1) : T(0) for j in 1:NC]
    pe2_data   = T[j == 2 ? T(1) : T(0) for j in 1:NC]

    ExaModels.@add_par(c, podd,  podd_data)
    ExaModels.@add_par(c, peven, peven_data)
    ExaModels.@add_par(c, peNC1, peNC1_data)
    ExaModels.@add_par(c, pe2,   pe2_data)

    ExaModels.@add_con(c,
        podd[j]  * (x[j]^2 * x[j+3] + sin(x[j+3] - x[j+4] + peNC1[j] * (x[j+4] - x[2])) - T(1)) +
        peven[j] * (x[j] + pe2[j] * (x[NC+3] - x[j]) + x[j+1]^2 * x[j+2] - T(2))
        for j in 1:NC;
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
