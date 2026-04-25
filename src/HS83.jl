function hs83_model(; T = Float64, backend = nothing, kwargs...)
    # CUTEst ordering: x[1]=X1, x[2]=X5, x[3]=X3, x[4]=X4, x[5]=X2
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5;
                       lvar = T[78.0, 27.0, 27.0, 27.0, 33.0],
                       uvar = T[102.0, 45.0, 45.0, 45.0, 45.0],
                       start = T[78.0, 27.0, 27.0, 27.0, 33.0])
    # OBJ = 37.293239*X1 + 5.3578547*X3² + 0.8356891*X1*X5 - 40792.141
    ExaModels.@add_obj(c,
        T(37.293239)*x[1] + T(5.3578547)*x[3]^2 + T(0.8356891)*x[1]*x[2] - T(40792.141))
    # C1 (G, RANGE 92): A2*X2*X5 + A3*X1*X4 - A4*X3*X5 + 85.334407 ∈ [0, 92]
    ExaModels.@add_con(c,
        T(0.0056858)*x[5]*x[2] + T(0.0006262)*x[1]*x[4] - T(0.0022053)*x[3]*x[2] + T(85.334407);
        lcon=T(0), ucon=T(92))
    # C2 (G, RANGE 20): A6*X2*X5 + A7*X1*X2 + A8*X3² - 9.48751 ∈ [0, 20]
    ExaModels.@add_con(c,
        T(0.0071317)*x[5]*x[2] + T(0.0029955)*x[1]*x[5] + T(0.0021813)*x[3]^2 - T(9.48751);
        lcon=T(0), ucon=T(20))
    # C3 (G, RANGE 5): A10*X3*X5 + A11*X1*X3 + A12*X3*X4 - 10.699039 ∈ [0, 5]
    ExaModels.@add_con(c,
        T(0.0047026)*x[3]*x[2] + T(0.0012547)*x[1]*x[3] + T(0.0019085)*x[3]*x[4] - T(10.699039);
        lcon=T(0), ucon=T(5))
    return ExaModels.ExaModel(c; kwargs...)
end
