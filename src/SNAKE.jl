# SNAKE: Nonconvex feasibility with sine-curve boundaries (Toint 1993)
# 2 free variables: X, Y; start = [1, 5]
# Objective: X
# CON1 (L ≤ 0): -Y + sin(X) ≤ 0
# CON2 (G ≥ 0): -Y + TIP*X + sin(X) ≥ 0, TIP = 0.0001
function snake_model(; T = Float64, backend = nothing, kwargs...)
    TIP = T(1) / T(10000)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(1), T(5)])
    ExaModels.@add_obj(c, x[1])
    ExaModels.@add_con(c, -x[2] + sin(x[1]); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, -x[2] + TIP*x[1] + sin(x[1]); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
