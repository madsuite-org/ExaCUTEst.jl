# DENSCHNF: Dennis-Schnabel example F (2 variables, unconstrained)
# Source: Dennis & Schnabel, "Numerical Methods for Unconstrained Optimization", p.107
# SIF: 2 groups (A,B), all L2 type, all N type, all variables FREE, start=(2,0)
#   A: E1=ISQP(U=x1+x2, U^2) coeff 2, E2=ISQM(U=x1-x2, U^2) coeff 1, CONSTANT=8
#      v_A = 2*(x1+x2)^2 + (x1-x2)^2 - 8  → contribution = v_A^2
#   B: E3=SSQ(x1,S=0)=x1^2 coeff 5, E4=SSQ(x2,S=3)=(x2-3)^2 coeff 1, CONSTANT=9
#      v_B = 5*x1^2 + (x2-3)^2 - 9  → contribution = v_B^2
function denschnf_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T[2.0, 0.0])
    ExaModels.@add_obj(c, (T(2) * (x[1] + x[2])^2 + (x[1] - x[2])^2 - T(8))^2)
    ExaModels.@add_obj(c, (T(5) * x[1]^2 + (x[2] - T(3))^2 - T(9))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
