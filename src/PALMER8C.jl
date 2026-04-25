function palmer8c_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # Variables: A0, A2, A4, A6, A8, A10, A12, A14 (all free); start = 1.0
    ExaModels.@add_var(c, x, 8;
        lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf],
        uvar = T[Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf],
        start = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # X data (radians), M=23, loop from index 12 to 23 (12 points)
    ExaModels.@add_par(c, XI, T[
        0.000000, 0.174533, 0.314159, 0.436332, 0.514504, 0.610865,
        0.785398, 0.959931, 1.134464, 1.308997, 1.483530, 1.570796])
    # Y data (kJ/mol)
    ExaModels.@add_par(c, YI, T[
         4.757534,  3.121416,  1.207606,  0.131916,  0.000000,  0.258514,
         3.380161, 10.762813, 23.745996, 44.471864, 76.541947, 97.874528])
    # model: A0 + A2*X^2 + A4*X^4 + A6*X^6 + A8*X^8 + A10*X^10 + A12*X^12 + A14*X^14
    ExaModels.@add_obj(c,
        (x[1] + x[2]*XI[i]^2 + x[3]*XI[i]^4 + x[4]*XI[i]^6 + x[5]*XI[i]^8 +
         x[6]*XI[i]^10 + x[7]*XI[i]^12 + x[8]*XI[i]^14 - YI[i])^2
        for i in 1:12)
    return ExaModels.ExaModel(c; kwargs...)
end
