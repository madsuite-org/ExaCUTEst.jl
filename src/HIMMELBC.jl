function himmelbc_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf,-Inf], uvar = T[Inf,Inf],
                       start = T[1.0, 1.0])
    ExaModels.@add_con(c, x[1]^2 + x[2] - T(11))
    ExaModels.@add_con(c, x[1] + x[2]^2 - T(7))
    return ExaModels.ExaModel(c; kwargs...)
end
