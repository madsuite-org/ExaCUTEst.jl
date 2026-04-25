function hs79_model(; T = Float64, backend = nothing, kwargs...)
    # CUTEst ordering: x[1]=X1, x[2]=X2, x[3]=X3, x[4]=X5, x[5]=X4
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5;
                       lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf],
                       start = T[2.0, 2.0, 2.0, 2.0, 2.0])
    ExaModels.@add_obj(c,
        (x[1]-x[2])^2 + (x[2]-x[3])^2 + (x[1]-1)^2 + (x[3]-x[5])^4 + (x[5]-x[4])^4)
    # C1: X1 + X2² + X3³ - (2+3*sqrt(2)) = x[1]+x[2]²+x[3]³-(2+3√2)
    ExaModels.@add_con(c, x[1] + x[2]^2 + x[3]^3 - T(2 + 3*sqrt(2)))
    # C2: X2 + X4 - X3² - (2√2-2) → x[2]+x[5]-x[3]²-(2√2-2)
    ExaModels.@add_con(c, x[2] + x[5] - x[3]^2 - T(2*sqrt(2) - 2))
    # C3: X1*X5 - 2 → x[1]*x[4] - 2
    ExaModels.@add_con(c, x[1]*x[4] - 2)
    return ExaModels.ExaModel(c; kwargs...)
end
