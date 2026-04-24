function hs37_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[0.0, 0.0, 0.0], uvar = T[42.0, 42.0, 42.0],
                       start = T[10.0, 10.0, 10.0])
    ExaModels.@add_obj(c, -x[1] * x[2] * x[3])
    ExaModels.@add_con(c, T(72) - x[1] - T(2)*x[2] - T(2)*x[3]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[1] + T(2)*x[2] + T(2)*x[3]; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
