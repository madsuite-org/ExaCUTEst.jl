function hs98_model(; T = Float64, backend = nothing, kwargs...)
    # CUTEst ordering: x[1]=X1, x[2]=X6, x[3]=X3, x[4]=X4, x[5]=X5, x[6]=X2
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 6;
                       lvar = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       uvar = T[0.31, 0.0134, 0.068, 0.042, 0.028, 0.046],
                       start = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ExaModels.@add_obj(c,
        T(4.3)*x[1] + T(4.7)*x[2] + T(63.3)*x[3] + T(15.8)*x[4] + T(68.5)*x[5] + T(31.8)*x[6])
    ExaModels.@add_con(c,
        T(17.1)*x[1] + T(1495.5)*x[2] + T(204.2)*x[3] + T(212.3)*x[4] + T(623.4)*x[5] + T(38.2)*x[6]
        - T(169)*x[1]*x[3] - T(3580)*x[3]*x[5] - T(3810)*x[4]*x[5] - T(18500)*x[4]*x[2] - T(24300)*x[5]*x[2]
        - T(32.97);
        lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c,
        T(17.9)*x[1] + T(1385.2)*x[2] + T(113.9)*x[3] + T(169.7)*x[4] + T(337.8)*x[5] + T(36.8)*x[6]
        - T(139)*x[1]*x[3] - T(2450)*x[4]*x[5] - T(16600)*x[4]*x[2] - T(17200)*x[5]*x[2]
        - T(25.12);
        lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c,
        -T(273)*x[6] - T(70)*x[4] - T(819)*x[5] + T(26000)*x[4]*x[5] + T(124.08);
        lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c,
        T(159.9)*x[1] - T(311)*x[6] + T(587)*x[4] + T(391)*x[5] + T(2198)*x[2]
        - T(14000)*x[1]*x[2] + T(173.02);
        lcon=T(0), ucon=T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
