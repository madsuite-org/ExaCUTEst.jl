function bqp1var_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # G1 (N type, identity): obj += X1
    # G2 (N type, GROUP TYPE L2): obj += X1^2
    ExaModels.@add_var(c, x, 1; lvar = T[0.0], uvar = T[0.5], start = T[0.25])
    ExaModels.@add_obj(c, x[1] + x[1]^2)
    return ExaModels.ExaModel(c; kwargs...)
end
