function hatfldcne_model(; T = Float64, backend = nothing, kwargs...)
    N = 25
    lvar = fill(T(0.0), N)
    uvar = fill(T(10.0), N)
    lvar[N] = T(-Inf)
    uvar[N] = T(Inf)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = T(0.9))
    ExaModels.@add_con(c, x[1] - T(1))
    ExaModels.@add_con(c, x[d+1] - x[d]^2 for d in 2:N-1)
    ExaModels.@add_con(c, x[N] - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
