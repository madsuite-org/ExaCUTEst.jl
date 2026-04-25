# PFIT1: Pole-fitting model with derivative constraints (Toint 1993)
# 3 variables: A (x[1]) free, R (x[2]) free, H (x[3]) ≥ -0.5; start = [1,0,1]
# 3 equality constraints: EF, EG, EH using compound element types T1-T5
# T1 = A*R*H; T3 = A*(A+1)*R*H²; T4 = R*(1-(1+H)^(-A)); T5 = A*(A+1)*R*H²*(1-(1+H)^(-(A+2)))
# T2 = A*R*H*(1-(1+H)^(-(A+1)))
# EF: -0.5*T3 + T1 - T4 = CF = -8
# EG: -T3 + T2 = CG = -18.666...
# EH: -T5 = CH = -23.111...
function pfit1_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[T(-Inf), T(-Inf), T(-0.5)],
        uvar = T(Inf), start = T[T(1), T(0), T(1)])
    # EF: -0.5*A*(A+1)*R*H² + A*R*H - R*(1-(1+H)^(-A)) + 8 = 0
    ExaModels.@add_con(c,
        - T(0.5)*x[1]*(x[1]+T(1))*x[2]*x[3]^2
        + x[1]*x[2]*x[3]
        - x[2]*(T(1) - exp(-x[1]*log(T(1)+x[3])))
        + T(8))
    # EG: -A*(A+1)*R*H² + A*R*H*(1-(1+H)^(-(A+1))) + 18.666... = 0
    ExaModels.@add_con(c,
        - x[1]*(x[1]+T(1))*x[2]*x[3]^2
        + x[1]*x[2]*x[3]*(T(1) - exp(-(x[1]+T(1))*log(T(1)+x[3])))
        + T(18.6666666666))
    # EH: -A*(A+1)*R*H²*(1-(1+H)^(-(A+2))) + 23.111... = 0
    ExaModels.@add_con(c,
        - x[1]*(x[1]+T(1))*x[2]*x[3]^2*(T(1) - exp(-(x[1]+T(2))*log(T(1)+x[3])))
        + T(23.1111111111))
    return ExaModels.ExaModel(c; kwargs...)
end
