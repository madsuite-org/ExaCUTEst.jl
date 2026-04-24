function hs57_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[0.4, -4.0], uvar = T[Inf, Inf],
                       start = T[0.42, 5.0])
    ExaModels.@add_par(c, A, T[8,8,10,10,10,10,12,12,12,12,14,14,14,16,16,16,
                                18,18,20,20,20,22,22,22,24,24,24,26,26,26,28,28,
                                30,30,30,32,32,34,36,36,38,38,40,42])
    ExaModels.@add_par(c, B, T[0.49,0.49,0.48,0.47,0.48,0.47,0.46,0.46,0.45,0.43,
                                0.45,0.43,0.43,0.44,0.43,0.43,0.46,0.45,0.42,0.42,
                                0.43,0.41,0.41,0.40,0.42,0.40,0.40,0.41,0.40,0.41,
                                0.41,0.40,0.40,0.40,0.38,0.41,0.40,0.40,0.41,0.38,
                                0.40,0.40,0.39,0.39])
    # OBSQ element: R = B_i - x1 - (0.49-x1)*exp(-x2*(A_i-8)); F = R^2
    ExaModels.@add_obj(c,
        (B[i] - x[1] - (T(0.49) - x[1]) * exp(-x[2] * (A[i] - 8)))^2 for i in 1:44)
    # CON1 (G): 0.49*x2 - x1*x2 - 0.09 >= 0
    ExaModels.@add_con(c, T(0.49)*x[2] - x[1]*x[2] - T(0.09); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
