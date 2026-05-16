# PRICE3NE: Nonlinear equations version of PRICE3 (2 variables, 2 equality constraints)
# Source: SCIPY global optimization benchmark (Price03)
# F1: (x1^2 - x2) / 0.1 = 10*(x1^2 - x2) = 0  (SCALE=0.1)
# F2: 6.4*(x2 - 0.5)^2 - x1 - 0.6 = 0
# All variables free; start: x1=1, x2=5
# SIF: Nick Gould, Jan 2020; classification NOR2-MN-2-2
function price3ne_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
                       start = T[1.0, 5.0])
    # F1: SCALE=0.1, element SQR(x1)=x1^2, linear -x2; residual = (x1^2 - x2)/0.1
    ExaModels.@add_con(c, T(10) * (x[1]^2 - x[2]))
    # F2: element 6.4*SSQR(x2)=6.4*(x2-0.5)^2, linear -x1, constant 0.6
    ExaModels.@add_con(c, T(6.4) * (x[2] - T(0.5))^2 - x[1] - T(0.6))
    return ExaModels.ExaModel(c; kwargs...)
end
