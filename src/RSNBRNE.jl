function rsnbrne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf,-Inf], uvar = T[Inf,Inf],
                       start = T[-1.2, 1.0])
    # G1 (E, scale=0.1): v=(x[2]-x[1]²), residual=v/0.1=10*(x[2]-x[1]²)
    # G2 (E, no scale):  v=x[1]-1
    ExaModels.@add_con(c, x[1] - T(1))
    ExaModels.@add_con(c, (x[2] - x[1]^2) / T(0.1))
    return ExaModels.ExaModel(c; kwargs...)
end
