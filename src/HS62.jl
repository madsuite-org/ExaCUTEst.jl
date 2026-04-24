function hs62_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
                       lvar = T[0.0, 0.0, 0.0],
                       uvar = T[1.0, 1.0, 1.0],
                       start = T[0.7, 0.2, 0.1])
    # LN3N: log(x1+x2+x3+0.03), LN3D: log(0.09x1+x2+x3+0.03)
    # LN2N: log(x2+x3+0.03), LN2D: log(0.07x2+x3+0.03)
    # LN1N: log(x3+0.03), LN1D: log(0.13x3+0.03)
    ExaModels.@add_obj(c,
        T(-8204.37)*log(x[1]+x[2]+x[3]+T(0.03)) +
        T(8204.37)*log(T(0.09)*x[1]+x[2]+x[3]+T(0.03)) +
        T(-9008.72)*log(x[2]+x[3]+T(0.03)) +
        T(9008.72)*log(T(0.07)*x[2]+x[3]+T(0.03)) +
        T(-9330.46)*log(x[3]+T(0.03)) +
        T(9330.46)*log(T(0.13)*x[3]+T(0.03)))
    ExaModels.@add_con(c, x[1] + x[2] + x[3] - 1)
    return ExaModels.ExaModel(c; kwargs...)
end
