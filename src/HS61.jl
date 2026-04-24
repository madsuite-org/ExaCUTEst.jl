function hs61_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # CUTEst ordering: x[1]=X3, x[2]=X2, x[3]=X1
    ExaModels.@add_var(c, x, 3;
                       lvar = T[-Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf],
                       start = T[0.0, 0.0, 0.0])
    # OBJ: -24*X3 + 16*X2 - 33*X1 + 2*X3² + 2*X2² + 4*X1²
    ExaModels.@add_obj(c, -24*x[1] + 16*x[2] - 33*x[3] + 2*x[1]^2 + 2*x[2]^2 + 4*x[3]^2)
    # C1: 3*X1 - 2*X2² - 7 = 0
    ExaModels.@add_con(c, 3*x[3] - 2*x[2]^2 - 7)
    # C2: 4*X1 - X3² - 11 = 0
    ExaModels.@add_con(c, 4*x[3] - x[1]^2 - 11)
    return ExaModels.ExaModel(c; kwargs...)
end
