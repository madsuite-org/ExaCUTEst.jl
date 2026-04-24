function hs72_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
                       lvar = T[0.001, 0.001, 0.001, 0.001],
                       uvar = T[4e5, 3e5, 2e5, 1e5],
                       start = T[1.0, 1.0, 1.0, 1.0])
    # OBJ: x1+x2+x3+x4 + 1  (CONSTANT = -1 → subtract -1 = add 1)
    ExaModels.@add_obj(c, x[1] + x[2] + x[3] + x[4] + 1)
    # CON1 (G): -4/x1 - 2.25/x2 - 1/x3 - 0.25/x4 + 0.0401 ≥ 0
    ExaModels.@add_con(c,
        -4/x[1] - T(2.25)/x[2] - 1/x[3] - T(0.25)/x[4] + T(0.0401);
        lcon=T(0), ucon=T(Inf))
    # CON2 (G): -0.16/x1 - 0.36/x2 - 0.64/x3 - 0.64/x4 + 0.010085 ≥ 0
    ExaModels.@add_con(c,
        -T(0.16)/x[1] - T(0.36)/x[2] - T(0.64)/x[3] - T(0.64)/x[4] + T(0.010085);
        lcon=T(0), ucon=T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
