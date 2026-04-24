function hs4_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[1.0, 0.0], uvar = T[Inf, Inf],
                       start = T[1.125, 0.125])
    # G1: CUBE type, scale=3, v=x[1]+1  → v³/σ = (x[1]+1)³/3
    # G2: identity, v=x[2]
    ExaModels.@add_obj(c, (x[1] + T(1))^3 / T(3) + x[2])
    return ExaModels.ExaModel(c; kwargs...)
end
