# BIGGS6: Biggs EXP problem in 6 variables (unconstrained, M=13 groups)
# minimize sum_{i=1}^{13} v_i^2
# v_i = X3*exp(-0.1i*X1) - X4*exp(-0.1i*X2) + X6*exp(-0.1i*X5) - Y(i)
# Y(i) = exp(-0.1i) - 5*exp(-i) + 3*exp(-0.4i)
# BOUNDS: FR DEFAULT → all variables free
function biggs6_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 6;
                       lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf, Inf],
                       start = T[1.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    # Precomputed constants Y(i) = exp(-0.1i) - 5*exp(-i) + 3*exp(-0.4i)
    ExaModels.@add_par(c, Y, T[exp(-0.1*i) - 5*exp(-float(i)) + 3*exp(-0.4*i) for i in 1:13])

    # v_i = X3*exp(-0.1i*X1) - X4*exp(-0.1i*X2) + X6*exp(-0.1i*X5) - Y(i)
    # contribution = v_i^2 (L2 group type, no scale)
    ExaModels.@add_obj(c,
        (x[3] * exp(T(-0.1) * i * x[1]) - x[4] * exp(T(-0.1) * i * x[2]) +
         x[6] * exp(T(-0.1) * i * x[5]) - Y[i])^2
        for i in 1:13)

    return ExaModels.ExaModel(c; kwargs...)
end
