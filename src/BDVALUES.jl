# BDVALUES: Boundary value problem with starting point scaled by X0SCALE=1000
# NDP=10002 variables; same structure as BDVALUE with X0SCALE applied to starting point.
# CUTEst ordering (rotated, same as BDVALUE):
#   pos 1 = X(NDP-1), pos j = X(j) for j=2..NDP-2
#   pos NDP-1 = X(1) fixed at 0, pos NDP = X(NDP) fixed at 0
function bdvalues_model(NDP = 10002; T = Float64, backend = nothing, kwargs...)
    H = T(1) / T(NDP - 1)
    halfH2 = T(0.5) * H * H
    X0SCALE = T(1000)
    lvar = fill(T(-Inf), NDP)
    uvar = fill(T(Inf), NDP)
    lvar[NDP - 1] = uvar[NDP - 1] = T(0)
    lvar[NDP]     = uvar[NDP]     = T(0)

    start = Vector{T}(undef, NDP)
    start[1] = X0SCALE * T(NDP - 2) * H * (T(NDP - 2) * H - T(1))  # X(NDP-1)
    for j in 2:NDP-2
        start[j] = X0SCALE * T(j - 1) * H * (T(j - 1) * H - T(1))  # X(j)
    end
    start[NDP - 1] = T(0)
    start[NDP]     = T(0)

    b_mid = T[T(i) * H + T(1) for i in 3:NDP-3]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NDP; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, BMID, b_mid)

    # G(2): X(1)→pos NDP-1, X(2)→pos 2, X(3)→pos 3
    ExaModels.@add_con(c,
        -x[NDP - 1] + T(2) * x[2] - x[3] +
        halfH2 * (x[2] + T(2) * H + T(1))^3)

    # G(3..NDP-3): X(I)→pos I for I in 3..NDP-3
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
