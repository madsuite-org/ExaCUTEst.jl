# MAKELA1: nonlinear minmax problem in two variables (Makela 1990)
# 3 variables: x1, x2, U (all free)
# minimize U
# subject to:
#   F1 (L): -U - x1 - x2 >= 0
#   F2 (L): -U - x1 - x2 + x1^2 + x2^2 - 1 >= 0
# Start: x1=-0.5, x2=-0.5, U=0.0
function makela1_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
                       start = T[-0.5, -0.5, 0.0])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c, -x[3] - x[1] - x[2]; lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, -x[3] - x[1] - x[2] + x[1]^2 + x[2]^2 - T(1); lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
