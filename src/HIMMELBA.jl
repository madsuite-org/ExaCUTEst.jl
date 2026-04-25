# HIMMELBA: Himmelblau problem 25 (2 free vars, start=[8,9])
# Two equality constraints (E-type groups, identity function, scale ignored for identity):
#   G1: X1 - 5 = 0 (scale=0.25 doesn't affect contribution for identity type)
#   G2: X2 - 6 = 0
# No objective (no N-type groups)
function himmelba_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T[T(8), T(9)])
    ExaModels.@add_con(c, x[1] - T(5))
    ExaModels.@add_con(c, x[2] - T(6))
    return ExaModels.ExaModel(c; kwargs...)
end
