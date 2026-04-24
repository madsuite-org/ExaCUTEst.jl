function himmelbd_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf,-Inf], uvar = T[Inf,Inf],
                       start = T[1.0, 1.0])
    ExaModels.@add_con(c, x[1]^2 + T(12)*x[2] - T(1))
    ExaModels.@add_con(c, T(49)*x[1]^2 + T(49)*x[2]^2 + T(84)*x[1] + T(2324)*x[2] - T(681))
    return ExaModels.ExaModel(c; kwargs...)
end
