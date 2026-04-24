function bt1_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf,-Inf], uvar = T[Inf,Inf],
                       start = T[0.08, 0.06])
    # OBJ (N, identity): -x1 + 100*x1² + 100*x2² - 100
    ExaModels.@add_obj(c, -x[1] + T(100)*x[1]^2 + T(100)*x[2]^2 - T(100))
    # CON1 (E): x1² + x2² - 1 = 0
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
