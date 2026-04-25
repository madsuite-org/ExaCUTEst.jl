# SIPOW1: Semi-infinite LP, minimize x2 in unit circle (Powell 1992)
# 2 free variables: X1, X2; start = [0.8, 0.5]
# Objective: X2
# M=2000 constraints C(J) ≥ 0: cos(2πJ/M)*X1 + sin(2πJ/M)*X2 + 1 ≥ 0
function sipow1_model(M = 2000; T = Float64, backend = nothing, kwargs...)
    twopi = T(2) * T(pi)
    data = [(c = cos(twopi*j/M), s = sin(twopi*j/M)) for j in 1:M]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(0.8), T(0.5)])
    ExaModels.@add_obj(c, x[2])
    ExaModels.@add_con(c, d.c*x[1] + d.s*x[2] + T(1)
        for d in data; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
