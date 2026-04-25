# PFIT1LS: Least-squares version of PFIT1 (Toint 1994)
# 3 variables: A (x[1]) free, R (x[2]) free, H (x[3]) ≥ -0.5; start = [1,0,1]
# Objective: EF²+EG²+EH² where EF,EG,EH are the PFIT1 constraint residuals (L2 group type)
function pfit1ls_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[T(-Inf), T(-Inf), T(-0.5)],
        uvar = T(Inf), start = T[T(1), T(0), T(1)])
    ExaModels.@add_obj(c,
        (- T(0.5)*x[1]*(x[1]+T(1))*x[2]*x[3]^2
         + x[1]*x[2]*x[3]
         - x[2]*(T(1) - exp(-x[1]*log(T(1)+x[3])))
         + T(8))^2)
    ExaModels.@add_obj(c,
        (- x[1]*(x[1]+T(1))*x[2]*x[3]^2
         + x[1]*x[2]*x[3]*(T(1) - exp(-(x[1]+T(1))*log(T(1)+x[3])))
         + T(18.6666666666))^2)
    ExaModels.@add_obj(c,
        (- x[1]*(x[1]+T(1))*x[2]*x[3]^2*(T(1) - exp(-(x[1]+T(2))*log(T(1)+x[3])))
         + T(23.1111111111))^2)
    return ExaModels.ExaModel(c; kwargs...)
end
