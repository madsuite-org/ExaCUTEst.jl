function hs14_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[2.0, 2.0])
    ExaModels.@add_obj(c, (x[1] - T(2))^2 + (x[2] - T(1))^2)
    # CUTEst order: CON2 (E) before CON1 (G)
    ExaModels.@add_con(c, x[1] - T(2)*x[2] + T(1))
    ExaModels.@add_con(c, -T(0.25)*x[1]^2 - x[2]^2 + T(1); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
