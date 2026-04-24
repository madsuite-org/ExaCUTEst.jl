function hs26_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf, -Inf, -Inf], uvar = T[Inf, Inf, Inf],
                       start = T[-2.6, 2.0, 2.0])
    # OBJ: (x[1]-x[2])² + (x[2]-x[3])⁴
    ExaModels.@add_obj(c, (x[1] - x[2])^2 + (x[2] - x[3])^4)
    # CON1 (E): (1+x[2]²)*x[1] + x[3]⁴ - 3 = 0
    ExaModels.@add_con(c, (T(1) + x[2]^2) * x[1] + x[3]^4 - T(3))
    return ExaModels.ExaModel(c; kwargs...)
end
