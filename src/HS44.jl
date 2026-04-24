function hs44_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T[0.0, 0.0, 0.0, 0.0],
                       uvar = T[Inf, Inf, Inf, Inf],
                       start = T[0.0, 0.0, 0.0, 0.0])
    # obj = x1 - x2 - x3 - x1*x3 + x1*x4 - x2*x4 + x2*x3
    ExaModels.@add_obj(c,
        x[1] - x[2] - x[3] -
        x[1]*x[3] + x[1]*x[4] - x[2]*x[4] + x[2]*x[3])
    # 6 linear G-type constraints
    ExaModels.@add_con(c, -x[1] - T(2)*x[2] + T(8);  lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -T(4)*x[1] - x[2] + T(12); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -T(3)*x[1] - T(4)*x[2] + T(12); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -T(2)*x[3] - x[4] + T(8);  lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[3] - T(2)*x[4] + T(8);  lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -x[3] - x[4] + T(5);       lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
