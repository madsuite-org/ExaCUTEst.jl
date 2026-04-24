function hs23_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-50.0, -50.0], uvar = T[50.0, 50.0],
                       start = T[3.0, 1.0])
    ExaModels.@add_obj(c, x[1]^2 + x[2]^2)
    # CON1 (G, linear): x[1]+x[2] ≥ 1
    ExaModels.@add_con(c, x[1] + x[2] - T(1); lcon = T(0), ucon = T(Inf))
    # CON2 (G): x[1]²+x[2]² ≥ 1
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - T(1); lcon = T(0), ucon = T(Inf))
    # CON3 (G): 9x[1]²+x[2]² ≥ 9
    ExaModels.@add_con(c, T(9)*x[1]^2 + x[2]^2 - T(9); lcon = T(0), ucon = T(Inf))
    # CON4 (G): x[1]²-x[2] ≥ 0
    ExaModels.@add_con(c, x[1]^2 - x[2]; lcon = T(0), ucon = T(Inf))
    # CON5 (G): x[2]²-x[1] ≥ 0
    ExaModels.@add_con(c, x[2]^2 - x[1]; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
