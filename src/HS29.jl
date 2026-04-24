function hs29_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf, -Inf, -Inf], uvar = T[Inf, Inf, Inf],
                       start = T[1.0, 1.0, 1.0])
    # PROD3 element: F = -V1*V2*V3
    ExaModels.@add_obj(c, -x[1] * x[2] * x[3])
    # CON1 (G): -x[1]²-2x[2]²-4x[3]²+48 ≥ 0
    ExaModels.@add_con(c, -x[1]^2 - T(2)*x[2]^2 - T(4)*x[3]^2 + T(48); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
