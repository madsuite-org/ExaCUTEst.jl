# FMINSURF: Minimum surface problem with average height objective, P=75 default (5625 variables)
# Same as FMINSRF2 but with AVH = (sum x(I,J))^2/P^4 instead of center height penalty
function fminsurf_model(P = 75; T = Float64, backend = nothing, kwargs...)
    NV   = P * P
    PM1  = P - 1
    SCALE  = T(1) / T(PM1)^2
    P4     = T(P)^4

    X_idx(i, j) = (j - 1) * P + i

    H00    = T(1)
    SLOPEJ = T(4)
    SLOPEI = T(8)
    WTOE   = SLOPEJ / T(PM1)
    STON   = SLOPEI / T(PM1)
    H01    = H00 + SLOPEJ
    H10    = H00 + SLOPEI

    x0 = zeros(T, NV)
    for j in 1:P
        x0[X_idx(1, j)] = H00 + T(j - 1) * WTOE
        x0[X_idx(P, j)] = H10 + T(j - 1) * WTOE
    end
    for i in 2:(P - 1)
        x0[X_idx(i, 1)] = H00 + T(i - 1) * STON
        x0[X_idx(i, P)] = H01 + T(i - 1) * STON
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # S(I,J) groups: same as FMINSRF2
    sq_data = [(
        x11 = X_idx(i, j),
        x22 = X_idx(i + 1, j + 1),
        x21 = X_idx(i + 1, j),
        x12 = X_idx(i, j + 1),
    ) for i in 1:PM1 for j in 1:PM1]

    ExaModels.@add_obj(c,
        sqrt(T(0.5) * (x[d.x11] - x[d.x22])^2 + T(0.5) * (x[d.x21] - x[d.x12])^2 + SCALE)
        for d in sq_data)

    # AVH group: L2 type with SCALE P^4: (sum x(I,J))^2 / P^4
    # This is a single group involving all N variables
    avh_data = [(idx = [X_idx(i, j) for i in 1:P for j in 1:P],)]
    ExaModels.@add_obj(c,
        sum(x[d.idx[k]] for k in 1:NV)^2 / P4
        for d in avh_data)

    return ExaModels.ExaModel(c; kwargs...)
end
