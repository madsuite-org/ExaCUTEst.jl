function hs76_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
                       lvar = T[0.0, 0.0, 0.0, 0.0],
                       uvar = T[Inf, Inf, Inf, Inf],
                       start = T[0.5, 0.5, 0.5, 0.5])
    ExaModels.@add_obj(c,
        -x[1] - 3*x[2] + x[3] - x[4] +
        x[1]^2 + T(0.5)*x[2]^2 + x[3]^2 + T(0.5)*x[4]^2 -
        x[1]*x[3] + x[3]*x[4])
    # C1 (L ≤ 0): x1+2x2+x3+x4 - 5 ≤ 0
    ExaModels.@add_con(c, x[1] + 2*x[2] + x[3] + x[4] - 5; lcon=T(-Inf), ucon=T(0))
    # C2 (L ≤ 0): 3x1+x2+2x3-x4 - 4 ≤ 0
    ExaModels.@add_con(c, 3*x[1] + x[2] + 2*x[3] - x[4] - 4; lcon=T(-Inf), ucon=T(0))
    # C3 (G ≥ 0): x2+4x3 - 1.5 ≥ 0
    ExaModels.@add_con(c, x[2] + 4*x[3] - T(1.5); lcon=T(0), ucon=T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
