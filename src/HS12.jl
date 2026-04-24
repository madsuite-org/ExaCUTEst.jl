function hs12_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[0.0, 0.0])
    # OBJ: -7x[1]-7x[2]+0.5x[1]²+x[2]²-x[1]x[2]
    ExaModels.@add_obj(c, -T(7)*x[1] - T(7)*x[2] + T(0.5)*x[1]^2 + x[2]^2 - x[1]*x[2])
    # CON1 (G): -4x[1]²-x[2]²+25 ≥ 0
    ExaModels.@add_con(c, -T(4)*x[1]^2 - x[2]^2 + T(25); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
