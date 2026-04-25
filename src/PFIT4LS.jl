# PFIT4LS: Least-squares pole-fitting model variant 4 (Toint 1994)
function pfit4ls_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[T(-Inf), T(-Inf), T(-0.5)],
        uvar = T(Inf), start = T[T(1), T(0), T(1)])
    ExaModels.@add_obj(c,
        (- T(0.5)*x[1]*(x[1]+T(1))*x[2]*x[3]^2
         + x[1]*x[2]*x[3]
         - x[2]*(T(1) - exp(-x[1]*log(T(1)+x[3])))
         + T(98.9629629629))^2)
    ExaModels.@add_obj(c,
        (- x[1]*(x[1]+T(1))*x[2]*x[3]^2
         + x[1]*x[2]*x[3]*(T(1) - exp(-(x[1]+T(1))*log(T(1)+x[3])))
         + T(216.098765432))^2)
    ExaModels.@add_obj(c,
        (- x[1]*(x[1]+T(1))*x[2]*x[3]^2*(T(1) - exp(-(x[1]+T(2))*log(T(1)+x[3])))
         + T(239.670781893))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
