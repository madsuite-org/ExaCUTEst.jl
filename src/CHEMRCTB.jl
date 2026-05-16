# CHEMRCTB: Simplified tubular chemical reactor (N variables, N constraints)
# Default N=5000. Only temperature T(I), concentration = (B+1-T)/B.
# PE = PEM = PEH = 5 (equal Peclet numbers, adiabatic)
# Element ARR(T; G, BPLUS1) = exp(G - G/T)*(BPLUS1 - T) where G=GAMMA=25, B+1=1.5
# Constraint GT(I): temperature equation = 0
# T(I) >= 1e-7; start = 1.0
function chemrctb_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    PE   = T(5.0)
    D    = T(0.135)
    B    = T(0.5)
    GAMA = T(25.0)
    BPLUS1 = B + T(1.0)

    H = T(1) / T(N - 1)

    # Temperature coefficients (same formula as CHEMRCTA temperature part)
    one_H2PE = T(1) / (H * H * PE)
    HPE_val  = PE * H

    CTI_m1 = one_H2PE + T(1) / H
    CTI    = -T(2) * one_H2PE - T(1) / H
    CTI_p1 = one_H2PE
    CT1    = T(1) - HPE_val

    rhs_GT1 = -(PE * H)

    lvar = fill(T(1e-7), N)
    uvar = fill(T(Inf), N)
    start = ones(T, N)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # GT(1): -T(1) + CT1*T(2) - rhs = 0
    ExaModels.@add_con(c, -x[1] + CT1 * x[2] - rhs_GT1)

    # GT(I) for I=2..N-1: CTI_m1*T(I-1) + CTI*T(I) + CTI_p1*T(I+1) + D*ARR = 0
    # ARR = exp(G - G/T)*(B+1 - T)
    inner = [(i = i,) for i in 2:N-1]
    ExaModels.@add_con(c,
        CTI_m1 * x[d.i - 1] + CTI * x[d.i] + CTI_p1 * x[d.i + 1] +
        D * exp(GAMA - GAMA / x[d.i]) * (BPLUS1 - x[d.i])
        for d in inner)

    # GT(N): -T(N-1) + T(N) = 0
    ExaModels.@add_con(c, -x[N - 1] + x[N])

    return ExaModels.ExaModel(c; kwargs...)
end
