# EXPFITA: One-sided rational approximation to exp, R=11 points (OLR2, 5 vars, 22 constraints)
# Variables: P0,P1,P2,Q1,Q2 (indices 1-5), all free
# Objective: sum_{I=1}^{R} ((P0+P1*T+P2*T^2)/(exp(T)*(1+Q1*(T-5)+Q2*(T-5)^2)) - 1)^2
# Constraints:
#   C(I): P0+P1*T+P2*T^2 - Q1*(T-5)*exp(T) - Q2*(T-5)^2*exp(T) - exp(T) >= 0
#   B(I): Q1*(T-5) + Q2*(T-5)^2 + 0.99999 >= 0
# T(I) = (I-1)*5/(R-1) for I=1..R
function expfita_model(R = 11; T = Float64, backend = nothing, kwargs...)
    NV = 5
    x0 = T[1, 1, 6, 0, 0]

    # Precompute T(I) and exp(T(I))
    step = T(5) / T(R - 1)
    ti   = [(i - 1) * step for i in 1:R]
    eti  = [exp(ti[i]) for i in 1:R]

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)
    # Variables: x[1]=P0, x[2]=P1, x[3]=P2, x[4]=Q1, x[5]=Q2

    # Objective: sum_I ((P0+P1*T+P2*T^2) / (exp(T)*(1+Q1*(T-5)+Q2*(T-5)^2)) - 1)^2
    fit_data = [(ti = ti[i], eti = eti[i]) for i in 1:R]
    ExaModels.@add_obj(c,
        ((x[1] + x[2] * d.ti + x[3] * d.ti^2) /
         (d.eti * (T(1) + x[4] * (d.ti - T(5)) + x[5] * (d.ti - T(5))^2)) - T(1))^2
        for d in fit_data)

    # Constraints interleaved C(I), B(I) for I=1..R (CUTEst ordering: loop creates C(I) then B(I) per I)
    for i in 1:R
        tm5   = ti[i] - T(5)
        tm5sq = (ti[i] - T(5))^2
        # C(I): P0+P1*T+P2*T^2 - Q1*(T-5)*ET - Q2*(T-5)^2*ET - ET >= 0
        ExaModels.@add_con(c,
            x[1] + x[2] * ti[i] + x[3] * ti[i]^2 -
            x[4] * tm5 * eti[i] - x[5] * tm5sq * eti[i] - eti[i];
            lcon = T(0), ucon = T(Inf))
        # B(I): Q1*(T-5) + Q2*(T-5)^2 + 0.99999 >= 0
        ExaModels.@add_con(c,
            x[4] * tm5 + x[5] * tm5sq + T(0.99999);
            lcon = T(0), ucon = T(Inf))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
