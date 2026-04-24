function hs63_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
                       lvar = T[0.0, 0.0, 0.0],
                       uvar = T[Inf, Inf, Inf],
                       start = T[2.0, 2.0, 2.0])
    # OBJ: -x1²-2x2²-x3²-x1*(x2+x3)+1000  (CONSTANT=-1000 → -(-1000)=+1000)
    ExaModels.@add_obj(c, -x[1]^2 - 2*x[2]^2 - x[3]^2 - x[1]*(x[2]+x[3]) + 1000)
    # C1 (E): 8x1+14x2+7x3 = 56
    ExaModels.@add_con(c, 8*x[1] + 14*x[2] + 7*x[3] - 56)
    # C2 (E): x1²+x2²+x3² = 25
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 + x[3]^2 - 25)
    return ExaModels.ExaModel(c; kwargs...)
end
