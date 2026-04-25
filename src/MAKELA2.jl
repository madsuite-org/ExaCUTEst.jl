# MAKELA2: nonlinear minmax problem in two variables (Makela 1990)
# 3 variables: x1, x2, U (all free)
# minimize U
# subject to (L type: group value >= 0):
#   F1: -U + x1^2 + x2^2 >= 0
#   F2: -U - 40*x1 - 10*x2 + x1^2 + x2^2 - (-40) >= 0
#       = -U - 40*x1 - 10*x2 + x1^2 + x2^2 + 40 >= 0
#   F3: -U - 10*x1 - 20*x2 + x1^2 + x2^2 - (-60) >= 0
#       = -U - 10*x1 - 20*x2 + x1^2 + x2^2 + 60 >= 0
# Start: x1=-1.0, x2=5.0, U=0.0
function makela2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
                       start = T[-1.0, 5.0, 0.0])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c, -x[3] + x[1]^2 + x[2]^2; lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, -x[3] - T(40)*x[1] - T(10)*x[2] + x[1]^2 + x[2]^2 + T(40); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, -x[3] - T(10)*x[1] - T(20)*x[2] + x[1]^2 + x[2]^2 + T(60); lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
