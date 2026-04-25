# BOX3: Box problem in 3 variables (all variables free)
# minimize sum_{i=1}^{M} v_i^2
# v_i = exp(-0.1*i*X1) - exp(-0.1*i*X2) + C[i]*X3
# C[i] = exp(-i) - exp(-0.1*i)
# BOUNDS: FR DEFAULT (all free)
function box3_model(M = 100; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
                       lvar = T[-Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf],
                       start = T[0.0, 10.0, 1.0])

    # TI[i] = -0.1*i (parameter T in element EXPT)
    ExaModels.@add_par(c, TI, T[-0.1 * i for i in 1:M])
    # C[i] = exp(-i) - exp(-0.1*i)  (COEFF in GROUPS = MEMTI + EMRI)
    ExaModels.@add_par(c, C, T[exp(-float(i)) - exp(-0.1 * i) for i in 1:M])

    # v_i = exp(TI[i]*X1) - exp(TI[i]*X2) + C[i]*X3
    # contribution = v_i^2  (L2 group type, no scale)
    ExaModels.@add_obj(c,
        (exp(TI[i] * x[1]) - exp(TI[i] * x[2]) + C[i] * x[3])^2
        for i in 1:M)

    return ExaModels.ExaModel(c; kwargs...)
end
