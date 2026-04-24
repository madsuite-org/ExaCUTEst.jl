function hs30_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[1.0, -10.0, -10.0], uvar = T[10.0, 10.0, 10.0],
                       start = T[1.0, 1.0, 1.0])
    ExaModels.@add_obj(c, x[1]^2 + x[2]^2 + x[3]^2)
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - T(1); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
