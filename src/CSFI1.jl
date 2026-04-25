# CSFI1: Steel caster optimization (5 vars, 1 obj, 4 constraints)
# Variables: x[1]=THICK(lvar=7), x[2]=WID(lvar=0), x[3]=LEN(0≤≤60), x[4]=TPH(lvar=0), x[5]=IPM(lvar=0)
# Objective: -TPH = -x[4]
# CIPM: -x[5] + 117.370892*x[4]/(x[2]*x[1]) = 0
# CLEN: -x[3] + x[1]^2*x[5]/48 = 0
# WOT:  x[2]/x[1] - 2 ≤ 0
# TTW:  x[1]*x[2] - 200 in [0, 50]  →  lcon=0, ucon=50
function csfi1_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    lvar = T[T(7), T(0), T(0), T(0), T(0)]
    uvar = T[T(Inf), T(Inf), T(60), T(Inf), T(Inf)]
    ExaModels.@add_var(c, x, 5; lvar = lvar, uvar = uvar, start = T(0.5))
    ExaModels.@add_obj(c, -x[4])
    ExaModels.@add_con(c, -x[5] + T(117.370892) * x[4] / (x[2] * x[1]))
    ExaModels.@add_con(c, -x[3] + x[1]^2 * x[5] / T(48))
    ExaModels.@add_con(c, x[2] / x[1] - T(2); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, x[1] * x[2] - T(200); lcon = T(0), ucon = T(50))
    return ExaModels.ExaModel(c; kwargs...)
end
