# EXPFITC: One-sided rational approximation to exp, R=251 points (OLR2, 5 vars, 502 constraints)
# Same as EXPFITA but with R=251 fitting points
function expfitc_model(R = 251; T = Float64, backend = nothing, kwargs...)
    NV = 5
    x0 = T[1, 1, 6, 0, 0]

    step = T(5) / T(R - 1)
    ti   = [(i - 1) * step for i in 1:R]
    eti  = [exp(ti[i]) for i in 1:R]

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    fit_data = [(ti = ti[i], eti = eti[i]) for i in 1:R]
    ExaModels.@add_obj(c,
        ((x[1] + x[2] * d.ti + x[3] * d.ti^2) /
         (d.eti * (T(1) + x[4] * (d.ti - T(5)) + x[5] * (d.ti - T(5))^2)) - T(1))^2
        for d in fit_data)

    for i in 1:R
        tm5   = ti[i] - T(5)
        tm5sq = (ti[i] - T(5))^2
        ExaModels.@add_con(c,
            x[1] + x[2] * ti[i] + x[3] * ti[i]^2 -
            x[4] * tm5 * eti[i] - x[5] * tm5sq * eti[i] - eti[i];
            lcon = T(0), ucon = T(Inf))
        ExaModels.@add_con(c,
            x[4] * tm5 + x[5] * tm5sq + T(0.99999);
            lcon = T(0), ucon = T(Inf))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
