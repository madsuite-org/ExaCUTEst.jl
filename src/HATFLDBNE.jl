function hatfldbne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
                       lvar = T[1e-7, 1e-7, 1e-7, 1e-7],
                       uvar = T[Inf, 0.8, Inf, Inf],
                       start = T(0.1))
    ExaModels.@add_con(c, x[1] - T(1))
    ExaModels.@add_con(c, x[d-1] - sqrt(x[d]) for d in 2:4)
    return ExaModels.ExaModel(c; kwargs...)
end
