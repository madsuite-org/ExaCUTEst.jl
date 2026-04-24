function hs45_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T[0.0, 0.0, 0.0, 0.0, 0.0],
                       uvar = T[1.0, 2.0, 3.0, 4.0, 5.0],
                       start = T[2.0, 2.0, 2.0, 2.0, 2.0])
    ExaModels.@add_obj(c, T(2) - x[1]*x[2]*x[3]*x[4]*x[5] / T(120))
    return ExaModels.ExaModel(c; kwargs...)
end
