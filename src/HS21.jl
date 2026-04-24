function hs21_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2;
                       lvar = T[2.0, -50.0], uvar = T[50.0, 50.0],
                       start = T[-1.0, -1.0])
    # OBJ: 0.01*x1² + x2² - 100
    ExaModels.@add_obj(c, T(0.01)*x[1]^2 + x[2]^2 - T(100))
    # CON1 (G type, ≥ 0): 10*x1 - x2 - 10 ≥ 0
    ExaModels.@add_con(c, T(10)*x[1] - x[2] - T(10); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
