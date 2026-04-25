function palmer5d_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # Variables: A0, A2, A4, A6 (all free); start = 1.0
    ExaModels.@add_var(c, x, 4;
        lvar = T[-Inf, -Inf, -Inf, -Inf],
        uvar = T[Inf, Inf, Inf, Inf],
        start = T[1.0, 1.0, 1.0, 1.0])
    # X data (radians), M=23, loop from index 12 to 23 (12 points)
    ExaModels.@add_par(c, XI, T[
        0.000000, 1.570796, 1.396263, 1.308997, 1.221730, 1.125835,
        1.047198, 0.872665, 0.698132, 0.523599, 0.349066, 0.174533])
    # Y data (kJ/mol)
    ExaModels.@add_par(c, YI, T[
        83.57418, 81.007654, 18.983286,  8.051067,  2.044762,  0.000000,
         1.170451, 10.479881, 25.785001, 44.126844, 62.822177, 77.719674])
    # model: A0 + A2*X^2 + A4*X^4 + A6*X^6
    ExaModels.@add_obj(c,
        (x[1] + x[2]*XI[i]^2 + x[3]*XI[i]^4 + x[4]*XI[i]^6 - YI[i])^2
        for i in 1:12)
    return ExaModels.ExaModel(c; kwargs...)
end
