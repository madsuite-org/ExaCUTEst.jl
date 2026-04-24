function hs20_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-0.5, -Inf], uvar = T[0.5, Inf],
                       start = T[-2.0, 1.0])
    ExaModels.@add_obj(c, T(100)*(x[2] - x[1]^2)^2 + (x[1] - T(1))^2)
    ExaModels.@add_con(c, x[1] + x[2]^2; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[2] + x[1]^2; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - T(1); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
