function hs64_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
                       lvar = T[0.00001, 0.00001, 0.00001],
                       uvar = T[Inf, Inf, Inf],
                       start = T[1.0, 1.0, 1.0])
    ExaModels.@add_obj(c, 5*x[1] + 20*x[2] + 10*x[3] +
                          50000/x[1] + 72000/x[2] + 144000/x[3])
    # CONSTR (L): 4/x1 + 32/x2 + 120/x3 ≤ 1
    ExaModels.@add_con(c, 4/x[1] + 32/x[2] + 120/x[3] - 1; lcon=T(-Inf), ucon=T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
