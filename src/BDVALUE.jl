# BDVALUE: Boundary value problem (More, Garbow, Hillstrom 1981, problem 28)
# NDP=5002 variables; X(1)=X(NDP)=0 fixed; NDP-2 free interior variables X(2..NDP-1)
#
# CUTEst ordering (FR DEFAULT + XX X(1) + XX X(NDP)):
#   When the first declared variable is fixed, sifdecoder rotates free variable positions:
#   the LAST free variable (X(NDP-1)) wraps to position 1, X(2..NDP-2) fill positions 2..NDP-2.
#   pos 1 = X(NDP-1), pos j = X(j) for j=2..NDP-2, pos NDP-1 = X(1) fixed, pos NDP = X(NDP) fixed.
#
# Constraint G(I) for I=2..NDP-1:
#   -X(I-1) + 2*X(I) - X(I+1) + 0.5*H^2*(X(I)+I*H+1)^3 = 0
# Start: X(I) = (I-1)*H*((I-1)*H - 1)
function bdvalue_model(NDP = 5002; T = Float64, backend = nothing, kwargs...)
    H = T(1) / T(NDP - 1)
    halfH2 = T(0.5) * H * H
    lvar = fill(T(-Inf), NDP)
    uvar = fill(T(Inf), NDP)
    lvar[NDP - 1] = uvar[NDP - 1] = T(0)   # X(1) fixed at pos NDP-1
    lvar[NDP]     = uvar[NDP]     = T(0)    # X(NDP) fixed at pos NDP

    # Start: X(I) starts at (I-1)*H*((I-1)*H - 1)
    # pos 1 = X(NDP-1): start = (NDP-2)*H*((NDP-2)*H - 1)
    # pos j = X(j) for j=2..NDP-2: start = (j-1)*H*((j-1)*H - 1)
    start = Vector{T}(undef, NDP)
    start[1] = T(NDP - 2) * H * (T(NDP - 2) * H - T(1))
    for j in 2:NDP-2
        start[j] = T(j - 1) * H * (T(j - 1) * H - T(1))
    end
    start[NDP - 1] = T(0)
    start[NDP]     = T(0)

    # B parameters for vectorised middle constraints G(3..NDP-3): BMID[k] = (k+2)*H+1
    b_mid = T[T(i) * H + T(1) for i in 3:NDP-3]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NDP; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, BMID, b_mid)

    # G(2): X(1)→pos NDP-1, X(2)→pos 2, X(3)→pos 3
    ExaModels.@add_con(c,
        -x[NDP - 1] + T(2) * x[2] - x[3] +
        halfH2 * (x[2] + T(2) * H + T(1))^3)

    # G(3..NDP-3): X(I)→pos I, X(I±1)→pos I±1 (all in 2..NDP-2)
    ExaModels.@add_con(c,
        -x[i - 1] + T(2) * x[i] - x[i + 1] +
        halfH2 * (x[i] + BMID[i - 2])^3
        for i in 3:(NDP - 3))

    # G(NDP-2): X(NDP-3)→pos NDP-3, X(NDP-2)→pos NDP-2, X(NDP-1)→pos 1
    ExaModels.@add_con(c,
        -x[NDP - 3] + T(2) * x[NDP - 2] - x[1] +
        halfH2 * (x[NDP - 2] + T(NDP - 2) * H + T(1))^3)

    # G(NDP-1): X(NDP-2)→pos NDP-2, X(NDP-1)→pos 1, X(NDP)→pos NDP
    ExaModels.@add_con(c,
        -x[NDP - 2] + T(2) * x[1] - x[NDP] +
        halfH2 * (x[1] + T(NDP - 1) * H + T(1))^3)

    return ExaModels.ExaModel(c; kwargs...)
end
