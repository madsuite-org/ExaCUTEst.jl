function hs34_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # XU 'DEFAULT' 100 → lvar=0, uvar=100; X3 additionally UP 10
    ExaModels.@add_var(c, x, 3; lvar = T[0.0, 0.0, 0.0], uvar = T[100.0, 100.0, 10.0],
                       start = T[0.0, 1.05, 2.9])
    ExaModels.@add_obj(c, -x[1])
    # CON1 (G): x2 - exp(x1) ≥ 0 ;  CON2 (G): x3 - exp(x2) ≥ 0
    ExaModels.@add_con(c, x[2] - exp(x[1]); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[3] - exp(x[2]); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
