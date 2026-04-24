function hs19_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[13.0, 0.0], uvar = T[100.0, 100.0],
                       start = T[20.1, 5.84])
    ExaModels.@add_obj(c, (x[1] - T(10))^3 + (x[2] - T(20))^3)
    ExaModels.@add_con(c, (x[1] - T(5))^2 + (x[2] - T(5))^2 - T(100); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, -(x[2] - T(5))^2 - (x[1] - T(6))^2 + T(82.81); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
