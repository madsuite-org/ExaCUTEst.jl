function hs46_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # CUTEst variable ordering: x[1]=X1, x[2]=X5, x[3]=X3, x[4]=X4, x[5]=X2
    ExaModels.@add_var(c, x, 5; lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf],
                       start = T[sqrt(2.0)*0.5, 2.0, 0.5, 2.0, 1.75])
    # OBJ: E1=(X1-X2)^2 + E2=(X3-1)^2 + E3=(X4-1)^4 + E4=(X5-1)^6
    ExaModels.@add_obj(c,
        (x[1] - x[5])^2 + (x[3] - 1)^2 + (x[4] - 1)^4 + (x[2] - 1)^6)
    # CON1 (E): X1^2*X4 + sin(X4-X5) = 1
    ExaModels.@add_con(c, x[1]^2 * x[4] + sin(x[4] - x[2]) - 1)
    # CON2 (E): X2 + X3^4*X4^2 = 2
    ExaModels.@add_con(c, x[5] + x[3]^4 * x[4]^2 - 2)
    return ExaModels.ExaModel(c; kwargs...)
end
