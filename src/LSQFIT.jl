function lsqfit_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # BOUNDS: a has lvar=0 (SIF default); b is free (XR → lvar=-Inf)
    ExaModels.@add_var(c, x, 2; lvar = T[0.0, -Inf], uvar = T[Inf, Inf],
                       start = T[0.0, 0.0])
    # 5 SQUARE groups (F=v², scale=2 → contribution=v²/2)
    # v_i = a*Xi + b - Yi;  pass data as ExaModels parameters
    c, Xi = ExaModels.add_par(c, T[0.1, 0.3, 0.5, 0.7, 0.9])
    c, Yi = ExaModels.add_par(c, T[0.25, 0.3, 0.625, 0.701, 1.0])
    ExaModels.@add_obj(c, (x[1]*Xi[i] + x[2] - Yi[i])^2 / T(2) for i in 1:5)
    # Cons (L type, ≤ 0): a + b - 0.85 ≤ 0
    ExaModels.@add_con(c, x[1] + x[2] - T(0.85); lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
