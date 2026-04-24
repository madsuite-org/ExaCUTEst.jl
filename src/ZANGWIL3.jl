function zangwil3_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf,-Inf,-Inf], uvar = T[Inf,Inf,Inf],
                       start = T[100.0, -1.0, 2.5])
    ExaModels.@add_con(c,  x[1] - x[2] + x[3])
    ExaModels.@add_con(c, -x[1] + x[2] + x[3])
    ExaModels.@add_con(c,  x[1] + x[2] - x[3])
    return ExaModels.ExaModel(c; kwargs...)
end
