# ALSOTAME: simple bound-constrained problem with nonlinear constraint (2 variables)
# minimize exp(x - 2*y)
# subject to  sin(-x + y - 1) = 0
# bounds: -2 <= x <= 2, -1.5 <= y <= 1.5
# Source: Conn, Gould, Toint, LANCELOT User's Manual (1991)
# SIF: Object=EXPN(x-2y), Constr=SINE(-x+y-1)=0
function alsotame_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-2.0, -1.5], uvar = T[2.0, 1.5])
    ExaModels.@add_obj(c, exp(x[1] - 2 * x[2]))
    ExaModels.@add_con(c, sin(-x[1] + x[2] - 1))
    return ExaModels.ExaModel(c; kwargs...)
end
