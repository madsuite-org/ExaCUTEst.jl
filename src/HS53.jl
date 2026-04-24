function hs53_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T[-10.0, -10.0, -10.0, -10.0, -10.0],
                       uvar = T[10.0, 10.0, 10.0, 10.0, 10.0],
                       start = T[2.0, 2.0, 2.0, 2.0, 2.0])
    ExaModels.@add_obj(c,
        (x[1] - x[2])^2 + (x[2] + x[3] - 2)^2 + (x[4] - 1)^2 + (x[5] - 1)^2)
    # No CONSTANTS section → no RHS subtraction
    ExaModels.@add_con(c, x[1] + 3*x[2])
    ExaModels.@add_con(c, x[3] + x[4] - 2*x[5])
    ExaModels.@add_con(c, x[2] - x[5])
    return ExaModels.ExaModel(c; kwargs...)
end
