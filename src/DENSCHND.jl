# DENSCHND: Dennis-Schnabel example D (3 variables, unconstrained)
# Source: Dennis & Schnabel, "Numerical Methods for Unconstrained Optimization", p.83
# SIF: 3 groups (A,B,C), all L2 type, all N type, all variables FREE, start=(10,10,10)
#   A: v_A = x1^2 + x2^3 - x3^4  (E1=SQ(x1), E2=CB(x2), E3=FR(x3) coeff -1)
#      contribution = v_A^2
#   B: v_B = 2*x1*x2*x3            (E4=3PR(x1,x2,x3) appears twice)
#      contribution = v_B^2
#   C: v_C = 2*x1*x2 - 3*x2*x3 + x1*x3  (E5=2PR(x1,x2) coeff 2, E6=2PR(x2,x3) coeff -3, E7=2PR(x1,x3))
#      contribution = v_C^2
function denschnd_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf), start = T(10.0))
    ExaModels.@add_obj(c, (x[1]^2 + x[2]^3 - x[3]^4)^2)
    ExaModels.@add_obj(c, (T(2) * x[1] * x[2] * x[3])^2)
    ExaModels.@add_obj(c, (T(2) * x[1] * x[2] - T(3) * x[2] * x[3] + x[1] * x[3])^2)
    return ExaModels.ExaModel(c; kwargs...)
end
