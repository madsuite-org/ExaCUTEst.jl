function hypcir_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf], start = T[0, 1])
    # G1: X1*X2 = 1
    ExaModels.@add_con(c, x[1] * x[2] - T(1))
    # G2: X1^2 + X2^2 = 4
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - T(4))
    return ExaModels.ExaModel(c; kwargs...)
end
