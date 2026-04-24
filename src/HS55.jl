function hs55_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # CUTEst variable ordering: x[1]=X1, x[2]=X4, x[3]=X3, x[4]=X2, x[5]=X5, x[6]=X6
    # UP bounds for X1 and X4 → x[1]≤1, x[2]≤1
    ExaModels.@add_var(c, x, 6;
                       lvar = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       uvar = T[1.0, 1.0, Inf, Inf, Inf, Inf],
                       start = T[1.0, 0.0, 0.0, 2.0, 0.0, 2.0])
    # OBJ: X1 + 2*X2 + 4*X5 + exp(X1*X4) = x1 + 2*x4 + 4*x5 + exp(x1*x2)
    ExaModels.@add_obj(c, x[1] + 2*x[4] + 4*x[5] + exp(x[1]*x[2]))
    ExaModels.@add_con(c, x[1] + 2*x[4] + 5*x[5] - 6)
    ExaModels.@add_con(c, x[1] + x[4] + x[3] - 3)
    ExaModels.@add_con(c, x[2] + x[5] + x[6] - 2)
    ExaModels.@add_con(c, x[1] + x[2] - 1)
    ExaModels.@add_con(c, x[4] + x[5] - 2)
    ExaModels.@add_con(c, x[3] + x[6] - 2)
    return ExaModels.ExaModel(c; kwargs...)
end
