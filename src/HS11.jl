function hs11_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[4.9, 0.1])
    ExaModels.@add_obj(c, (x[1] - T(5))^2 + x[2]^2 - T(25))
    ExaModels.@add_con(c, x[2] - x[1]^2; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
