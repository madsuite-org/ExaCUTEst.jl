function hs77_model(; T = Float64, backend = nothing, kwargs...)
    # CUTEst ordering: x[1]=X1, x[2]=X5, x[3]=X3, x[4]=X4, x[5]=X2
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5;
                       lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf],
                       start = T[2.0, 2.0, 2.0, 2.0, 2.0])
    # OBJ = (X1-1)² + (X1-X2)² + (X3-1)² + (X4-1)^4 + (X5-1)^6
    ExaModels.@add_obj(c,
        (x[1]-1)^2 + (x[1]-x[5])^2 + (x[3]-1)^2 + (x[4]-1)^4 + (x[2]-1)^6)
    # CON1: X1²*X4 + sin(X4-X5) - 2*sqrt(2) = x[1]²*x[4] + sin(x[4]-x[2]) - 2sqrt(2)
    ExaModels.@add_con(c, x[1]^2*x[4] + sin(x[4]-x[2]) - T(2*sqrt(2)))
    # CON2: X2 + X3^4*X4^2 - (sqrt(2)+8) = x[5] + x[3]^4*x[4]^2 - (sqrt(2)+8)
    ExaModels.@add_con(c, x[5] + x[3]^4*x[4]^2 - T(sqrt(2)+8))
    return ExaModels.ExaModel(c; kwargs...)
end
