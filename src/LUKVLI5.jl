# LUKVLI5: Inequality version of LUKVLE5 (G-type constraints ≥ 0)
# Source: Problem 5.5, Luksan & Vlcek (1999), TR 767
# Nvar=N+2 CUTEst variables, N-4 G-type constraints
# Identical to LUKVLE5 except constraints are ≥ 0 (lcon=0, ucon=Inf)
function lukvli5_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert N >= 7 "N must be >= 7"
    Nvar = N + 2

    start = fill(T(-1), Nvar)
    start[N+1] = T(0)
    start[N+2] = T(0)

    lvar = fill(T(-Inf), Nvar)
    uvar = fill(T(Inf), Nvar)
    lvar[N+1] = T(0); uvar[N+1] = T(0)
    lvar[N+2] = T(0); uvar[N+2] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, Nvar; lvar = lvar, uvar = uvar, start = start)

    ExaModels.@add_obj(c, abs(T(3)*x[N] - x[3] - x[N+1] - T(2)*x[N]^2 + T(1))^(T(7)/T(3)))
    ExaModels.@add_obj(c, abs(T(3)*x[3] - x[4] - x[N] - T(2)*x[3]^2 + T(1))^(T(7)/T(3)))
    ExaModels.@add_obj(c,
        abs(T(3)*x[i+1] - x[i+2] - x[i] - T(2)*x[i+1]^2 + T(1))^(T(7)/T(3))
        for i in 3:(N-3)
    )
    ExaModels.@add_obj(c, abs(T(3)*x[N-1] - x[2] - x[N-2] - T(2)*x[N-1]^2 + T(1))^(T(7)/T(3)))
    ExaModels.@add_obj(c, abs(T(3)*x[2] - x[1] - x[N-1] - T(2)*x[2]^2 + T(1))^(T(7)/T(3)))
    ExaModels.@add_obj(c, abs(T(3)*x[1] - x[N+2] - x[2] - T(2)*x[1]^2 + T(1))^(T(7)/T(3)))

    ExaModels.@add_con(c,
        T(6)*x[4] - x[N] + x[5] + T(8)*(x[4]^3 - x[4]*x[3]) -
        T(4)*x[5]^2 + x[3]^2 - x[6]^2 - T(2);
        lcon = T(0), ucon = T(Inf)
    )
    ExaModels.@add_con(c,
        T(6)*x[k+3] - x[k+1] + x[k+4] + T(8)*(x[k+3]^3 - x[k+3]*x[k+2]) -
        T(4)*x[k+4]^2 + x[k+2]^2 - x[k+5]^2 - T(2)
        for k in 2:(N-6);
        lcon = T(0), ucon = T(Inf)
    )
    ExaModels.@add_con(c,
        T(6)*x[N-2] - x[N-4] + x[N-1] + T(8)*(x[N-2]^3 - x[N-2]*x[N-3]) -
        T(4)*x[N-1]^2 + x[N-3]^2 - x[2]^2 - T(2);
        lcon = T(0), ucon = T(Inf)
    )
    ExaModels.@add_con(c,
        T(6)*x[N-1] - x[N-3] + x[2] + T(8)*(x[N-1]^3 - x[N-1]*x[N-2]) -
        T(4)*x[2]^2 + x[N-2]^2 - x[1]^2 - T(2);
        lcon = T(0), ucon = T(Inf)
    )

    return ExaModels.ExaModel(c; kwargs...)
end
