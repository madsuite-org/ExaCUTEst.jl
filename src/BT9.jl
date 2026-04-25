function bt9_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # 4 variables, all free, start = 2.0
    # CUTEst variable ordering: x[1]=X1, x[2]=X4, x[3]=X3, x[4]=X2
    ExaModels.@add_var(c, x, 4; lvar = fill(T(-Inf), 4), uvar = fill(T(Inf), 4),
                       start = fill(T(2.0), 4))
    # OBJ (N, identity): linear -X1 = -x[1]
    ExaModels.@add_obj(c, -x[1])
    # CON1 (E): X2 - X1^3 - X3^2 = x[4] - x[1]^3 - x[3]^2 = 0
    ExaModels.@add_con(c, x[4] - x[1]^3 - x[3]^2)
    # CON2 (E): -X2 + X1^2 - X4^2 = -x[4] + x[1]^2 - x[2]^2 = 0
    ExaModels.@add_con(c, -x[4] + x[1]^2 - x[2]^2)
    return ExaModels.ExaModel(c; kwargs...)
end
