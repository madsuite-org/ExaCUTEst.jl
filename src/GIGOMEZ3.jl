# GIGOMEZ3: Minimax problem (Gigola & Gomez 1990)
# 3 free variables: X1, X2, Z; start = [2, 2, 2]
# Objective: Z
# 3 inequality constraints (XG: ≥ 0):
#   C1: Z - X1^4 - X2^2 ≥ 0
#   C2: Z - (2-X1)^2 - (2-X2)^2 ≥ 0
#   C3: Z - 2*exp(X2-X1) ≥ 0
function gigomez3_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(2), T(2), T(2)])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c, x[3] - x[1]^4 - x[2]^2; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[3] - (T(2)-x[1])^2 - (T(2)-x[2])^2; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[3] - T(2)*exp(x[2]-x[1]); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
