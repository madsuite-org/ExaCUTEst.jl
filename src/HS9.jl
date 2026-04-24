function hs9_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf,-Inf], uvar = T[Inf,Inf],
                       start = T[0.0, 0.0])
    ExaModels.@add_obj(c, sin(T(π)*x[1]/T(12))*cos(T(π)*x[2]/T(16)))
    ExaModels.@add_con(c, T(4)*x[1] - T(3)*x[2])
    return ExaModels.ExaModel(c; kwargs...)
end
