function hs28_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf, -Inf, -Inf], uvar = T[Inf, Inf, Inf],
                       start = T[-4.0, 1.0, 1.0])
    # OBJ: (x[1]+x[2])² + (x[2]+x[3])²
    ExaModels.@add_obj(c, (x[1] + x[2])^2 + (x[2] + x[3])^2)
    # CON1 (E): x[1]+2x[2]+3x[3]-1 = 0
    ExaModels.@add_con(c, x[1] + T(2)*x[2] + T(3)*x[3] - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
