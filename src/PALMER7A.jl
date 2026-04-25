function palmer7a_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # Variables: A0, A2, A4, A6 (free), B >= 1e-5, C >= 1e-5; start = 1.0
    ExaModels.@add_var(c, x, 6;
        lvar = T[-Inf, -Inf, -Inf, -Inf, 1e-5, 1e-5],
        uvar = T[Inf, Inf, Inf, Inf, Inf, Inf],
        start = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # X data (radians), M=24, loop from index 12 to 24 (13 points)
    ExaModels.@add_par(c, XI, T[
        0.000000, 0.139626, 0.261799, 0.436332, 0.565245, 0.512942,
        0.610865, 0.785398, 0.959931, 1.134464, 1.308997, 1.483530,
        1.658063])
    # Y data (kJ/mol)
    ExaModels.@add_par(c, YI, T[
          4.419446,  3.564931,  2.139067,  0.404686,  0.000000,  0.035152,
          0.146813,  2.718058,  9.474417, 26.132221, 41.451561, 72.283164,
        117.630959])
    # model: A0 + A2*X^2 + A4*X^4 + A6*X^6 + B/(C + X^2)
    ExaModels.@add_obj(c,
        (x[1] + x[2]*XI[i]^2 + x[3]*XI[i]^4 + x[4]*XI[i]^6 + x[5]/(x[6] + XI[i]^2) - YI[i])^2
        for i in 1:13)
    return ExaModels.ExaModel(c; kwargs...)
end
