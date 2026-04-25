# PFIT3LS: Least-squares pole-fitting model variant 3 (Toint 1994)
function pfit3ls_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[T(-Inf), T(-Inf), T(-0.5)],
        uvar = T(Inf), start = T[T(1), T(0), T(1)])
    ExaModels.@add_obj(c,
        (- T(0.5)*x[1]*(x[1]+T(1))*x[2]*x[3]^2
         + x[1]*x[2]*x[3]
         - x[2]*(T(1) - exp(-x[1]*log(T(1)+x[3])))
         + T(56.8888888888))^2)
    ExaModels.@add_obj(c,
        (- x[1]*(x[1]+T(1))*x[2]*x[3]^2
         + x[1]*x[2]*x[3]*(T(1) - exp(-(x[1]+T(1))*log(T(1)+x[3])))
         + T(126.222222222))^2)
    ExaModels.@add_obj(c,
        (- x[1]*(x[1]+T(1))*x[2]*x[3]^2*(T(1) - exp(-(x[1]+T(2))*log(T(1)+x[3])))
         + T(143.407407407))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
