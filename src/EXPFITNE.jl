function expfitne_model(; T = Float64, backend = nothing, kwargs...)
    P = 10
    H = T(0.25)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; start = zeros(T, 2))
    data = [(ih = T(0.25) * T(i), rhs = T(0.25) * T(i)) for i in 1:P]
    ExaModels.@add_con(c, x[1] * exp(x[2] * d.ih) - d.rhs for d in data)
    return ExaModels.ExaModel(c; kwargs...)
end
