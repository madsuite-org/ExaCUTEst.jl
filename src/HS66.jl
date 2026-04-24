function hs66_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
                       lvar = T[0.0, 0.0, 0.0],
                       uvar = T[100.0, 100.0, 10.0],
                       start = T[0.0, 1.05, 2.9])
    ExaModels.@add_obj(c, -T(0.8)*x[1] + T(0.2)*x[3])
    # C1 (G): x2 - exp(x1) ≥ 0
    ExaModels.@add_con(c, x[2] - exp(x[1]); lcon=T(0), ucon=T(Inf))
    # C2 (G): x3 - exp(x2) ≥ 0
    ExaModels.@add_con(c, x[3] - exp(x[2]); lcon=T(0), ucon=T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
