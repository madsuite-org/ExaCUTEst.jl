function booth_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf,-Inf], uvar = T[Inf,Inf],
                       start = T[0.0, 0.0])
    ExaModels.@add_con(c, x[1] + T(2)*x[2] - T(7))
    ExaModels.@add_con(c, T(2)*x[1] + x[2] - T(5))
    return ExaModels.ExaModel(c; kwargs...)
end
