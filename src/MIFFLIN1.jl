# MIFFLIN1: Nonlinear minmax, 2 variables + slack U
# Source: M.M. Makela, Ph.D. thesis, 1990; SIF: Ph. Toint, Nov 1993
#
# Variables: X1=x[1], X2=x[2], U=x[3] (all free), start=[0.8,0.6,0.0]
# Objective: minimize U
# F1 (L type): X1^2 + X2^2 - U - X1 - 1 ≤ 0  (elements SQ on X1,X2; linear -U,-X1; const 1)
# F2 (L type): -U - X1 ≤ 0  (linear only; const 0)
function mifflin1_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar  = T[-Inf, -Inf, -Inf],
        uvar  = T[Inf, Inf, Inf],
        start = T[0.8, 0.6, 0.0])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - x[3] - x[1] - T(1); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, -x[3] - x[1];                           lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
