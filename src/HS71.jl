function hs71_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
                       lvar = T[1.0, 1.0, 1.0, 1.0],
                       uvar = T[5.0, 5.0, 5.0, 5.0],
                       start = T[1.0, 5.0, 5.0, 1.0])
    # OBJ: x3 + x1*x4*(x1+x2+x3)  (LP element: TX=x1, TY=x4, U=x1+x2+x3)
    ExaModels.@add_obj(c, x[3] + x[1]*x[4]*(x[1]+x[2]+x[3]))
    # CUTEst puts equality (C2) before inequality (C1)
    # C2 (E): x1²+x2²+x3²+x4² - 40 = 0
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 - 40)
    # C1 (G): x1*x2*x3*x4 - 25 ≥ 0
    ExaModels.@add_con(c, x[1]*x[2]*x[3]*x[4] - 25; lcon=T(0), ucon=T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
