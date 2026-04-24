function hs10_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[-10.0, 10.0])
    ExaModels.@add_obj(c, x[1] - x[2])
    # CON1 (G): -3x[1]²+2x[1]x[2]-x[2]²+1 ≥ 0
    ExaModels.@add_con(c, -T(3)*x[1]^2 + T(2)*x[1]*x[2] - x[2]^2 + T(1); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
