# DEMYMALO: Minimax problem (Demyanov & Malozemov 1974)
# 3 free variables: X1, X2, U; start = [1, 1, 0]
# Objective: U
# 3 inequality constraints (XL: ≤ 0):
#   F1: -U + 5*X1 + X2 ≤ 0
#   F2: -U - 5*X1 + X2 ≤ 0
#   F3: -U + 4*X2 + X1^2 + X2^2 ≤ 0
function demymalo_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(1), T(1), T(0)])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c, -x[3] + T(5)*x[1] + x[2]; lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, -x[3] - T(5)*x[1] + x[2]; lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, -x[3] + T(4)*x[2] + x[1]^2 + x[2]^2; lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
