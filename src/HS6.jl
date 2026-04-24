function hs6_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[-1.2, 1.0])
    # G1 (N, L2, scale=0.01): v = 1-x[1]; contribution = v²/0.01
    ExaModels.@add_obj(c, (x[1] - T(1))^2 / T(0.01))
    # G2 (E, scale=0.1): SQ element is -x[1]²; v = x[2]-x[1]²; residual = v/0.1
    ExaModels.@add_con(c, (x[2] - x[1]^2) / T(0.1))
    return ExaModels.ExaModel(c; kwargs...)
end
