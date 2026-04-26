# BDVALUES: Boundary value problem variant with scaled starting point
# NDP=10002 variables; same structure as BDVALUE but start scaled by X0SCALE=1000
# CUTEst ordering: x[j]=X(j+1) for j=1..NDP-2 (free), x[NDP-1]=X(1) fixed, x[NDP]=X(NDP) fixed
function bdvalues_model(NDP = 10002; T = Float64, backend = nothing, kwargs...)
    H = T(1) / T(NDP - 1)
    halfH2 = T(0.5) * H * H
    X0SCALE = T(1000)
    lvar = fill(T(-Inf), NDP)
    uvar = fill(T(Inf), NDP)
    lvar[NDP - 1] = uvar[NDP - 1] = T(0)
    lvar[NDP]     = uvar[NDP]     = T(0)
    start = Vector{T}(undef, NDP)
    for j in 1:NDP-2
        start[j] = X0SCALE * T(j) * H * (T(j) * H - T(1))
    end
    start[NDP - 1] = T(0)
    start[NDP]     = T(0)
    b_data = T[T(i) * H + T(1) for i in 2:NDP-1]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NDP; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, B, b_data)
    ExaModels.@add_con(c,
        -x[NDP - 1] + T(2) * x[1] - x[2] + halfH2 * (x[1] + T(2) * H + T(1))^3)
    ExaModels.@add_con(c,
        -x[j - 1] + T(2) * x[j] - x[j + 1] + halfH2 * (x[j] + B[j])^3
        for j in 2:NDP-3)
    ExaModels.@add_con(c,
        -x[NDP - 3] + T(2) * x[NDP - 2] - x[NDP] +
        halfH2 * (x[NDP - 2] + T(NDP - 1) * H + T(1))^3)
    return ExaModels.ExaModel(c; kwargs...)
end
