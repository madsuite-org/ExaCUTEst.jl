# DENSCHNE: Dennis-Schnabel example E (3 variables, unconstrained)
# Source: Dennis & Schnabel, "Numerical Methods for Unconstrained Optimization", p.107
# SIF: 3 groups (A,B,C), all L2 type, all N type, all variables FREE, start=(2,3,-8)
#   A: v_A = x1  (linear only)  → contribution = x1^2
#   B: v_B = x2 + x2^2  (linear x2 + element E2=SQ(x2))  → contribution = (x2 + x2^2)^2
#   C: v_C = exp(x3) - 1  (element E3=EX(x3), CONSTANT=1.0)  → contribution = (exp(x3)-1)^2
function denschne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf), start = T[2.0, 3.0, -8.0])
    ExaModels.@add_obj(c, x[1]^2)
    ExaModels.@add_obj(c, (x[2] + x[2]^2)^2)
    ExaModels.@add_obj(c, (exp(x[3]) - T(1))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
