function hs73_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
                       lvar = T[0.0, 0.0, 0.0, 0.0],
                       uvar = T[Inf, Inf, Inf, Inf],
                       start = T[1.0, 1.0, 1.0, 1.0])
    ExaModels.@add_obj(c, T(24.55)*x[1] + T(26.75)*x[2] + 39*x[3] + T(40.5)*x[4])
    # CUTEst ordering: C3 (linear equality) first, then C1 (linear G), then C2 (G with element)
    # C3 (E): x1+x2+x3+x4 - 1 = 0
    ExaModels.@add_con(c, x[1] + x[2] + x[3] + x[4] - 1)
    # C1 (G): 2.3x1+5.6x2+11.1x3+1.3x4 - 5 ≥ 0
    ExaModels.@add_con(c,
        T(2.3)*x[1] + T(5.6)*x[2] + T(11.1)*x[3] + T(1.3)*x[4] - 5;
        lcon=T(0), ucon=T(Inf))
    # C2 (G): 12x1+11.9x2+41.8x3+52.1x4 - 1.645*sqrt(0.28x1²+0.19x2²+20.5x3²+0.62x4²) - 21 ≥ 0
    ExaModels.@add_con(c,
        12*x[1] + T(11.9)*x[2] + T(41.8)*x[3] + T(52.1)*x[4] -
        T(1.645)*sqrt(T(0.28)*x[1]^2 + T(0.19)*x[2]^2 + T(20.5)*x[3]^2 + T(0.62)*x[4]^2) - 21;
        lcon=T(0), ucon=T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
