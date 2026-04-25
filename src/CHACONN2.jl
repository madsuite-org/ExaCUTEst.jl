# CHACONN2: Nonlinear minmax in 2 variables (Charalambous & Conn, SINUM 1978)
# 3 variables: X1, X2, U (all free), start = [2, 2, 0]
# Objective: minimize U
# 3 inequality constraints (XL: ≤ 0):
#   F1: -U + X1^4 + X2^2 ≤ 0
#   F2: -U + (2-X1)^2 + (2-X2)^2 ≤ 0
#   F3: -U + 2*exp(X2-X1) ≤ 0
function chaconn2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(2), T(2), T(0)])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c, -x[3] + x[1]^4 + x[2]^2; lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, -x[3] + (T(2) - x[1])^2 + (T(2) - x[2])^2; lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, -x[3] + T(2) * exp(x[2] - x[1]); lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
