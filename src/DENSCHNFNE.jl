function denschnfne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; start = T[2.0, 0.0])
    ExaModels.@add_con(c, T(2) * (x[1] + x[2])^2 + (x[1] - x[2])^2 - T(8))
    ExaModels.@add_con(c, T(5) * x[1]^2 + (x[2] - T(3))^2 - T(9))
    return ExaModels.ExaModel(c; kwargs...)
end
