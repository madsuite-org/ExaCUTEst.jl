function himmelbg_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[0.5, 0.5])
    # Element HG: F = exp(-X1-X2) * (2*X1^2 + 3*X2^2)
    # Group G is identity (no GROUP TYPE), no CONSTANT, no SCALE
    ExaModels.@add_obj(c, exp(-x[1] - x[2]) * (T(2) * x[1]^2 + T(3) * x[2]^2))
    return ExaModels.ExaModel(c; kwargs...)
end
