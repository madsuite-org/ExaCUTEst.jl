function byrdsphr_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
                       lvar = T[-Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf],
                       start = T[5.0, 0.0001, -0.0001])
    ExaModels.@add_obj(c, -x[1] - x[2] - x[3])
    # SPH1: x1² + x2² + x3² = 9
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 + x[3]^2 - 9)
    # SPH2: (x1-1)² + x2² + x3² = 9
    ExaModels.@add_con(c, (x[1]-1)^2 + x[2]^2 + x[3]^2 - 9)
    return ExaModels.ExaModel(c; kwargs...)
end
