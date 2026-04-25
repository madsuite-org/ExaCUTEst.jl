# MINMAXBD: Minmax version of Brown-Dennis problem (More' et al. 1981)
# 5 free variables: X1..X4, F; start = [25, 5, -5, -1, 825.559]
# Objective: F
# 20 inequality constraints G(I) ≤ 0:
#   -F + (X1 + (I/5)*X2 - exp(I/5))^2 + (X3 + sin(I/5)*X4 - cos(I/5))^2 ≤ 0
function minmaxbd_model(; T = Float64, backend = nothing, kwargs...)
    t = T.(1:20) ./ T(5)
    data = [(ci = t[i], ei = exp(t[i]), si = sin(t[i]), cs = cos(t[i])) for i in 1:20]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(25), T(5), T(-5), T(-1), T(825.559)])
    ExaModels.@add_obj(c, x[5])
    ExaModels.@add_con(c,
        -x[5] + (x[1] + d.ci*x[2] - d.ei)^2 + (x[3] + d.si*x[4] - d.cs)^2
        for d in data; lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
