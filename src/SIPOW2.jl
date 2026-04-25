# SIPOW2: Semi-infinite LP, symmetric variant (Powell 1992)
# 2 free variables: X1, X2; start = [0.8, 0.5]
# Objective: X2
# C(J) for J=1..M/2 ≥ 0: cos(4πJ/M)*X1 + sin(4πJ/M)*X2 + 1 ≥ 0
# C(J) for J=M/2+1..M ≥ 0: X1 + 1 ≥ 0 (M/2 copies)
function sipow2_model(M = 2000; T = Float64, backend = nothing, kwargs...)
    M2 = M ÷ 2
    fourpi = T(4) * T(pi)
    data1 = [(c = cos(fourpi*j/M), s = sin(fourpi*j/M)) for j in 1:M2]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(0.8), T(0.5)])
    ExaModels.@add_obj(c, x[2])
    ExaModels.@add_con(c, d.c*x[1] + d.s*x[2] + T(1)
        for d in data1; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[1] + T(1) for _ in 1:M2; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
