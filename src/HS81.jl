function hs81_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5;
                       lvar = T[-2.3, -2.3, -3.2, -3.2, -3.2],
                       uvar = T[2.3, 2.3, 3.2, 3.2, 3.2],
                       start = T[-2.0, 2.0, 2.0, -1.0, -1.0])
    # OBJ: exp(prod) - 0.5*x1^6 - 0.5*x2^6 - x1³*x2³ - x1³ - x2³ - 0.5
    #     = exp(x1*x2*x3*x4*x5) - 0.5*(1 + x1^3 + x2^3)^2
    ExaModels.@add_obj(c,
        exp(x[1]*x[2]*x[3]*x[4]*x[5]) -
        T(0.5)*x[1]^6 - T(0.5)*x[2]^6 - x[1]^3*x[2]^3 - x[1]^3 - x[2]^3 - T(0.5))
    # C1: x1²+x2²+x3²+x4²+x5² - 10 = 0
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2 - 10)
    # C2: x2*x3 - 5*x4*x5 = 0
    ExaModels.@add_con(c, x[2]*x[3] - 5*x[4]*x[5])
    # C3 (CONSTANT=-1): x1³+x2³+1 = 0
    ExaModels.@add_con(c, x[1]^3 + x[2]^3 + 1)
    return ExaModels.ExaModel(c; kwargs...)
end
