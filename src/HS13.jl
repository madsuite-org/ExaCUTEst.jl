function hs13_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[0.0, 0.0], uvar = T[Inf, Inf],
                       start = T[-2.0, -2.0])
    ExaModels.@add_obj(c, (x[1] - T(2))^2 + x[2]^2)
    # CON1 (G): (1-x[1])³-x[2] ≥ 0
    ExaModels.@add_con(c, (T(1) - x[1])^3 - x[2]; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
