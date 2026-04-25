# LUKVLI6: Inequality version of LUKVLE6 (G-type constraints ≥ 0)
# Source: Problem 5.6, Luksan & Vlcek (1999), TR 767
# N=9999 variables (default), N/2=4999 G-type constraints
# All variables free; start x[i]=3
function lukvli6_model(N = 9999; T = Float64, backend = nothing, kwargs...)
    N2 = N ÷ 2

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(3))

    ExaModels.@add_obj(
        c,
        abs(T(2) * x[1] + x[1] + x[2] + T(5) * x[1]^3 + x[1]^2 + x[2]^2 + T(1))^T(7 / 3)
    )
    ExaModels.@add_obj(
        c,
        abs(T(2) * x[2] + (x[1] + x[2] + x[3]) + T(5) * x[2]^3 +
            (x[1]^2 + x[2]^2 + x[3]^2) + T(1))^T(7 / 3)
    )
    if N >= 4
        ExaModels.@add_obj(
            c,
            abs(T(2) * x[3] + (x[1] + x[2] + x[3] + x[4]) + T(5) * x[3]^3 +
                (x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2) + T(1))^T(7 / 3)
        )
    end
    if N >= 5
        ExaModels.@add_obj(
            c,
            abs(T(2) * x[4] + (x[1] + x[2] + x[3] + x[4] + x[5]) + T(5) * x[4]^3 +
                (x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2) + T(1))^T(7 / 3)
        )
    end
    if N >= 6
        ExaModels.@add_obj(
            c,
            abs(T(2) * x[5] + (x[1] + x[2] + x[3] + x[4] + x[5] + x[6]) +
                T(5) * x[5]^3 +
                (x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2 + x[6]^2) +
                T(1))^T(7 / 3)
        )
    end
    if N >= 7
        ExaModels.@add_obj(
            c,
            abs(T(2) * x[i] +
                (x[i-5] + x[i-4] + x[i-3] + x[i-2] + x[i-1] + x[i] + x[i+1]) +
                T(5) * x[i]^3 +
                (x[i-5]^2 + x[i-4]^2 + x[i-3]^2 + x[i-2]^2 + x[i-1]^2 + x[i]^2 + x[i+1]^2) +
                T(1))^T(7 / 3) for i in 6:(N-1)
        )
    end
    ExaModels.@add_obj(
        c,
        abs(T(2) * x[N] +
            (x[N-5] + x[N-4] + x[N-3] + x[N-2] + x[N-1] + x[N]) +
            T(5) * x[N]^3 +
            (x[N-5]^2 + x[N-4]^2 + x[N-3]^2 + x[N-2]^2 + x[N-1]^2 + x[N]^2) +
            T(1))^T(7 / 3)
    )

    ExaModels.@add_con(
        c,
        T(4) * x[2k] - (x[2k-1] - x[2k+1]) * exp(x[2k-1] - x[2k+1] - x[2k]) - T(3)
        for k in 1:N2;
        lcon = T(0), ucon = T(Inf)
    )

    return ExaModels.ExaModel(c; kwargs...)
end
