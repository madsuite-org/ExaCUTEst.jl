function cb3_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # 3 variables: X1, X2, U — all free
    # Start: X1=2, X2=2, U=1
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
                       start = T[2.0, 2.0, 1.0])
    # OBJ (XN, identity): U = x[3]
    ExaModels.@add_obj(c, x[3])
    # C1 (XG, G type >=0): U - X1^4 - X2^2 >= 0
    ExaModels.@add_con(c, x[3] - x[1]^4 - x[2]^2; lcon = T(0), ucon = T(Inf))
    # C2 (XG, G type >=0): U - (2-X1)^2 - (2-X2)^2 >= 0
    ExaModels.@add_con(c, x[3] - (T(2) - x[1])^2 - (T(2) - x[2])^2; lcon = T(0), ucon = T(Inf))
    # C3 (XG, G type >=0): U - 2*exp(X2-X1) >= 0
    ExaModels.@add_con(c, x[3] - T(2)*exp(x[2] - x[1]); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
