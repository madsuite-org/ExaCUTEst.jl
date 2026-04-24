function hs47_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # CUTEst variable ordering: x[1]=X1, x[2]=X2, x[3]=X3, x[4]=X5, x[5]=X4
    ExaModels.@add_var(c, x, 5; lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf],
                       start = T[2.0, sqrt(2.0), -1.0, 0.5, 2.0 - sqrt(2.0)])
    # OBJ: (X1-X2)^2 + (X2-X3)^3 + (X3-X4)^4 + (X4-X5)^4
    ExaModels.@add_obj(c,
        (x[1] - x[2])^2 + (x[2] - x[3])^3 + (x[3] - x[5])^4 + (x[5] - x[4])^4)
    # CON1 (E): X1 + X2^2 + X3^3 = 3
    ExaModels.@add_con(c, x[1] + x[2]^2 + x[3]^3 - 3)
    # CON2 (E): X2 + X4 - X3^2 = 1  (X4=x[5])
    ExaModels.@add_con(c, x[2] + x[5] - x[3]^2 - 1)
    # CON3 (E): X1*X5 = 1  (X5=x[4])
    ExaModels.@add_con(c, x[1] * x[4] - 1)
    return ExaModels.ExaModel(c; kwargs...)
end
