# TAME: simple constrained linear least-squares (2 variables, 1 equality)
# minimize (x - y)^2
# subject to  x + y = 1
# Source: Conn, Gould, Toint, LANCELOT User's Manual (1991)
# SIF: Object=SQUARE(x-y), Constr=equality x+y-1=0
function tame_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # SIF default: variables are non-negative unless BOUNDS section says otherwise
    ExaModels.@add_var(c, x, 2; lvar = T[0.0, 0.0])
    ExaModels.@add_obj(c, (x[1] - x[2])^2)
    ExaModels.@add_con(c, x[1] + x[2] - 1)
    return ExaModels.ExaModel(c; kwargs...)
end
