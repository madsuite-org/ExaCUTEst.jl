function hs8_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[2.0, 1.0])
    # OBJ = -1 (constant); express as x[1]*0 - 1 to give ExaModels a variable reference
    ExaModels.@add_obj(c, x[1] * T(0) - T(1))
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - T(25))
    ExaModels.@add_con(c, x[1] * x[2] - T(9))
    return ExaModels.ExaModel(c; kwargs...)
end
