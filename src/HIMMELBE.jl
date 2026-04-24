function himmelbe_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf,-Inf,-Inf], uvar = T[Inf,Inf,Inf],
                       start = T[-1.2, 2.0, 0.0])
    # CUTEst constraint order: G3, G2, G1 (linear groups before nonlinear)
    ExaModels.@add_con(c, -x[2] + T(1))                          # G3: -X2+1
    ExaModels.@add_con(c, -x[1] + T(1))                          # G2: -X1+1
    ExaModels.@add_con(c, T(0.25)*(x[1]+x[2])^2 - x[3])         # G1: 0.25*(X1+X2)²-X3
    return ExaModels.ExaModel(c; kwargs...)
end
