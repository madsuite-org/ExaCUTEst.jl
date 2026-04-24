function hs27_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf, -Inf, -Inf], uvar = T[Inf, Inf, Inf],
                       start = T[2.0, 2.0, 2.0])
    # OBJ: 0.01*(1-x[1])² + (x[2]-x[1]²)²
    ExaModels.@add_obj(c, T(0.01) * (T(1) - x[1])^2 + (x[2] - x[1]^2)^2)
    # CON1 (E): x[1] + x[3]² + 1 = 0
    ExaModels.@add_con(c, x[1] + x[3]^2 + T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
