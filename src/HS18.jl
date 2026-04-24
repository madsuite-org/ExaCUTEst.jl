function hs18_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[2.0, 0.0], uvar = T[50.0, 50.0],
                       start = T[2.0, 2.0])
    ExaModels.@add_obj(c, T(0.01)*x[1]^2 + x[2]^2)
    ExaModels.@add_con(c, x[1]*x[2] - T(25); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - T(25); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
