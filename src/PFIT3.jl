# PFIT3: Pole-fitting model variant 3 (Toint 1993)
# Same structure as PFIT1 with CF=-56.888..., CG=-126.222..., CH=-143.407...
function pfit3_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[T(-Inf), T(-Inf), T(-0.5)],
        uvar = T(Inf), start = T[T(1), T(0), T(1)])
    ExaModels.@add_con(c,
        - T(0.5)*x[1]*(x[1]+T(1))*x[2]*x[3]^2
        + x[1]*x[2]*x[3]
        - x[2]*(T(1) - exp(-x[1]*log(T(1)+x[3])))
        + T(56.8888888888))
    ExaModels.@add_con(c,
        - x[1]*(x[1]+T(1))*x[2]*x[3]^2
        + x[1]*x[2]*x[3]*(T(1) - exp(-(x[1]+T(1))*log(T(1)+x[3])))
        + T(126.222222222))
    ExaModels.@add_con(c,
        - x[1]*(x[1]+T(1))*x[2]*x[3]^2*(T(1) - exp(-(x[1]+T(2))*log(T(1)+x[3])))
        + T(143.407407407))
    return ExaModels.ExaModel(c; kwargs...)
end
