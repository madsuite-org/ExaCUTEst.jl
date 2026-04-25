function denschna_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T[1.0, 1.0])
    ExaModels.@add_obj(c, x[1]^4)
    ExaModels.@add_obj(c, (x[1] + x[2])^2)
    ExaModels.@add_obj(c, (exp(x[2]) - T(1))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
