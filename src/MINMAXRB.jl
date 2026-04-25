# MINMAXRB: Rosenbrock-like minmax problem (Hald & Madsen 1981)
# 3 free variables: X1, X2, U; start = [-1.2, 1.0, 1.0]
# Objective: U
# 4 inequality constraints (XG: ≥ 0):
#   C1: U - 10*X2 + 10*X1^2 ≥ 0
#   C2: U + 10*X2 - 10*X1^2 ≥ 0
#   C3: U + X1 - 1 ≥ 0
#   C4: U - X1 + 1 ≥ 0
function minmaxrb_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(-1.2), T(1.0), T(1.0)])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c, x[3] - T(10)*x[2] + T(10)*x[1]^2; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[3] + T(10)*x[2] - T(10)*x[1]^2; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[3] + x[1] - T(1); lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[3] - x[1] + T(1); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
