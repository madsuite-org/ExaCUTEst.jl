function hs33_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[0.0, 0.0, 0.0], uvar = T[Inf, Inf, 5.0],
                       start = T[0.0, 0.0, 3.0])
    # OBJ: x[3] + (x[1]-1)*(x[1]-2)*(x[1]-3)
    ExaModels.@add_obj(c, x[3] + (x[1] - T(1)) * (x[1] - T(2)) * (x[1] - T(3)))
    # CON1 (G): x[3]²-x[2]²-x[1]² ≥ 0
    ExaModels.@add_con(c, x[3]^2 - x[2]^2 - x[1]^2; lcon = T(0), ucon = T(Inf))
    # CON2 (G): x[1]²+x[2]²+x[3]²-4 ≥ 0
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 + x[3]^2 - T(4); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
