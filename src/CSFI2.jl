# CSFI2: Steel caster optimization variant (5 vars, 1 obj, 4 constraints)
# Variables: x[1]=THICK(lvar=7), x[2]=WID(lvar=0), x[3]=LEN(lvar=0), x[4]=TPH(lvar=45), x[5]=IPM(lvar=0)
# Objective: LEN = x[3]
# Same element constraints as CSFI1 but minimizing LEN and TPH >= 45
function csfi2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    lvar = T[T(7), T(0), T(0), T(45), T(0)]
    uvar = T[T(Inf), T(Inf), T(Inf), T(Inf), T(Inf)]
    ExaModels.@add_var(c, x, 5; lvar = lvar, uvar = uvar, start = T(0.5))
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c, -x[5] + T(117.370892) * x[4] / (x[2] * x[1]))
    ExaModels.@add_con(c, -x[3] + x[1]^2 * x[5] / T(48))
    ExaModels.@add_con(c, x[2] / x[1] - T(2); lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c, x[1] * x[2] - T(200); lcon = T(0), ucon = T(50))
    return ExaModels.ExaModel(c; kwargs...)
end
