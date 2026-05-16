# POROUS2: Same as POROUS1 but with D = -50.0
function porous2_model(P = 72; T = Float64, backend = nothing, kwargs...)
    D = T(-50.0)
    H = T(1.0 / (P - 1))
    inv_H2 = T(1.0) / (H * H)
    D_2H = D / (T(2) * H)

    idx(i, j) = (j - 1) * P + i
    nvar = P * P

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)

    for j in 1:P
        lvar[idx(1, j)] = T(1.0); uvar[idx(1, j)] = T(1.0)
        lvar[idx(P, j)] = T(0.0); uvar[idx(P, j)] = T(0.0)
    end
    for i in 2:P-1
        lvar[idx(i, P)] = T(1.0); uvar[idx(i, P)] = T(1.0)
        lvar[idx(i, 1)] = T(0.0); uvar[idx(i, 1)] = T(0.0)
    end

    start = zeros(T, nvar)
    for i in 1:P
        for j in 1:P
            x1 = T(i - 1) * H
            x2 = T(j - 1) * H
            start[idx(i, j)] = T(1.0) - x1 * x2
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    ExaModels.@add_con(c,
        inv_H2 * (x[idx(i+1, j)]^2 + x[idx(i-1, j)]^2 + x[idx(i, j-1)]^2 + x[idx(i, j+1)]^2 - T(4) * x[idx(i, j)]^2) +
        D_2H * (x[idx(i+1, j)]^3 - x[idx(i-1, j)]^3)
        for i in 2:P-1, j in 2:P-1 if !(i == P-1 && j == P-1))

    let i = P - 1, j = P - 1
        ExaModels.@add_con(c,
            inv_H2 * (x[idx(i+1, j)]^2 + x[idx(i-1, j)]^2 + x[idx(i, j-1)]^2 + x[idx(i, j+1)]^2 - T(4) * x[idx(i, j)]^2) +
            D_2H * (x[idx(i+1, j)]^3 - x[idx(i-1, j)]^3) - T(-50.0))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
