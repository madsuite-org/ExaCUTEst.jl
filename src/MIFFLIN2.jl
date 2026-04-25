# MIFFLIN2: Nonlinear minmax, 2 variables + slack U
# Source: M.M. Makela, Ph.D. thesis, 1990; SIF: Ph. Toint, Nov 1993
#
# Variables: X1=x[1], X2=x[2], U=x[3] (all free), start=[-1,-1,0]
# Objective: minimize U
# F1 (L): 3.75*X1^2 + 3.75*X2^2 - U - X1 - 3.75 ≤ 0
# F2 (L): 0.25*X1^2 + 0.25*X2^2 - U - X1 - 0.25 ≤ 0
function mifflin2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar  = T[-Inf, -Inf, -Inf],
        uvar  = T[Inf, Inf, Inf],
        start = T[-1.0, -1.0, 0.0])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c, T(3.75) * x[1]^2 + T(3.75) * x[2]^2 - x[3] - x[1] - T(3.75);
        lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, T(0.25) * x[1]^2 + T(0.25) * x[2]^2 - x[3] - x[1] - T(0.25);
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
