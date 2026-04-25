function sisser_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf], start = T[1.0, 0.1])
    ExaModels.@add_obj(c, x[1]^4 / T(0.3333333) + x[1]^2 * x[2]^2 / T(0.5) + x[2]^4 / T(0.3333333))
    return ExaModels.ExaModel(c; kwargs...)
end
