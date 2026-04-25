# PFIT2LS: Least-squares pole-fitting model variant 2 (Toint 1994)
function pfit2ls_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[T(-Inf), T(-Inf), T(-0.5)],
        uvar = T(Inf), start = T[T(1), T(0), T(1)])
    ExaModels.@add_obj(c,
        (- T(0.5)*x[1]*(x[1]+T(1))*x[2]*x[3]^2
         + x[1]*x[2]*x[3]
         - x[2]*(T(1) - exp(-x[1]*log(T(1)+x[3])))
         + T(26.6666666666))^2)
    ExaModels.@add_obj(c,
        (- x[1]*(x[1]+T(1))*x[2]*x[3]^2
         + x[1]*x[2]*x[3]*(T(1) - exp(-(x[1]+T(1))*log(T(1)+x[3])))
         + T(60.4444444444))^2)
    ExaModels.@add_obj(c,
        (- x[1]*(x[1]+T(1))*x[2]*x[3]^2*(T(1) - exp(-(x[1]+T(2))*log(T(1)+x[3])))
         + T(71.1111111111))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
