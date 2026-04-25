# LUKVLI1: Inequality version of LUKVLE1 (L-type constraints ≤ 0)
# Source: Problem 5.1, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), N-2=9998 L-type constraints
# All variables free; start: odd→-1.2, even→1.0
function lukvli1_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    start = T[isodd(i) ? T(-1.2) : T(1.0) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    ExaModels.@add_obj(
        c,
        T(100) * (-x[i+1] + x[i]^2)^2 + (x[i] - T(1))^2 for i in 1:(N-1)
    )

    ExaModels.@add_con(
        c,
        T(3) * x[k+1]^3 + sin(x[k+1] - x[k+2]) * sin(x[k+1] + x[k+2]) -
        x[k] * exp(x[k] - x[k+1]) + T(2) * x[k+2] + T(4) * x[k+1] - T(8)
        for k in 1:(N-2);
        lcon = T(-Inf), ucon = T(0)
    )

    return ExaModels.ExaModel(c; kwargs...)
end
