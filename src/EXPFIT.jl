function expfit_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf,-Inf], uvar = T[Inf,Inf],
                       start = T[0.0, 0.0])
    # 10 L2 groups: v_i = x[1]*exp(0.25*i*x[2]) - 0.25*i, contribution = v_i²
    ExaModels.@add_obj(c, (x[1]*exp(T(0.25)*i*x[2]) - T(0.25)*i)^2 for i in 1:10)
    return ExaModels.ExaModel(c; kwargs...)
end
