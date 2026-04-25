# STANCMIN: Fractional program (Stancu-Minasian 1992)
# 3 variables (lvar=0): X1, X2, X3; start = 50
# Objective: -(3*X1+6*X2+2*X3-11)/(X1+4*X2+X3+1)
#   RATIO element: N = 3*X1+6*X2+2*X3, D = X1+4*X2+X3, F = -(N-11)/(D+1)
# 2 L-type constraints (≤ 0):
#   C1: 3*X1 + 4*X2 + X3 - 2 ≤ 0
#   C2: X1 + 4*X2 + X3 - 1 ≤ 0
function stancmin_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(0), uvar = T(Inf), start = T(50))
    ExaModels.@add_obj(
        c,
        -(T(3)*x[1] + T(6)*x[2] + T(2)*x[3] - T(11)) /
        (x[1] + T(4)*x[2] + x[3] + T(1)),
    )
    ExaModels.@add_con(c, T(3)*x[1] + T(4)*x[2] + x[3] - T(2); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, x[1] + T(4)*x[2] + x[3] - T(1); lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
