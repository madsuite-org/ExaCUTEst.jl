function hs84_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5;
                       lvar = T[0.0, 1.2, 20.0, 9.0, 6.5],
                       uvar = T[1000.0, 2.4, 60.0, 9.3, 7.0],
                       start = T[2.52, 2.0, 37.5, 9.25, 6.8])
    # OBJ = MA(2)*x1 + MA(3)*x1*x2 + MA(4)*x1*x3 + MA(5)*x1*x4 + MA(6)*x1*x5 - A(1)
    ExaModels.@add_obj(c,
        T(8720288.849)*x[1] - T(150512.5253)*x[1]*x[2] +
        T(156.6950325)*x[1]*x[3] - T(476470.3222)*x[1]*x[4] -
        T(729482.8271)*x[1]*x[5] + T(24345.0))
    # CON1 (G, RANGE 294000)
    ExaModels.@add_con(c,
        -T(145421.402)*x[1] + T(2931.1506)*x[1]*x[2] - T(40.427932)*x[1]*x[3] +
        T(5106.192)*x[1]*x[4] + T(15711.36)*x[1]*x[5];
        lcon=T(0), ucon=T(294000))
    # CON2 (G, RANGE 294000)
    ExaModels.@add_con(c,
        -T(155011.1084)*x[1] + T(4360.53352)*x[1]*x[2] + T(12.9492344)*x[1]*x[3] +
        T(10236.884)*x[1]*x[4] + T(13176.786)*x[1]*x[5];
        lcon=T(0), ucon=T(294000))
    # CON3 (G, RANGE 277200)
    ExaModels.@add_con(c,
        -T(326669.5104)*x[1] + T(7390.68412)*x[1]*x[2] - T(27.8986976)*x[1]*x[3] +
        T(16643.076)*x[1]*x[4] + T(30988.146)*x[1]*x[5];
        lcon=T(0), ucon=T(277200))
    return ExaModels.ExaModel(c; kwargs...)
end
