function hs117_model(; T = Float64, backend = nothing, kwargs...)
    # CUTEst ordering: x[1..5]=Y5,Y4,Y3,Y2,Y1  x[6..10]=X6..X10  x[11..15]=X5,X4,X3,X2,X1
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 15;
                       lvar = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0],
                       uvar = T[Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf,
                                Inf, Inf, Inf, Inf, Inf],
                       start = T[0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 60.0,
                                 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    # Linear in X: 40*X1+2*X2+0.25*X3+4*X4+4*X5+X6+40*X7+60*X8-5*X9-X10
    ExaModels.@add_obj(c, T(40)*x[15] + T(2)*x[14] + T(0.25)*x[13] + T(4)*x[12] + T(4)*x[11] +
                          x[6] + T(40)*x[7] + T(60)*x[8] - T(5)*x[9] - x[10])
    # Cubic in Y: 2*D_J*Y_J^3, D=[4,8,10,6,2] → Y1=x[5],Y2=x[4],Y3=x[3],Y4=x[2],Y5=x[1]
    ExaModels.@add_obj(c, T(8)*x[5]^3 + T(16)*x[4]^3 + T(20)*x[3]^3 + T(12)*x[2]^3 + T(4)*x[1]^3)
    # Quadratic in Y: sum_{K,J} C_{K,J}*Y_K*Y_J
    ExaModels.@add_obj(c,
        T(30)*x[5]^2 + T(39)*x[4]^2 + T(10)*x[3]^2 + T(39)*x[2]^2 + T(30)*x[1]^2 +
        T(-40)*x[5]*x[4] + T(-20)*x[5]*x[3] + T(64)*x[5]*x[2] + T(-20)*x[5]*x[1] +
        T(-12)*x[4]*x[3] + T(-62)*x[4]*x[2] + T(64)*x[4]*x[1] +
        T(-12)*x[3]*x[2] + T(-20)*x[3]*x[1] +
        T(-40)*x[2]*x[1])
    # C(1): lcon=0, G type
    ExaModels.@add_con(c,
        T(60)*x[5] + T(-40)*x[4] + T(-20)*x[3] + T(64)*x[2] + T(-20)*x[1] +
        T(16)*x[15] + T(3.5)*x[13] + T(-2)*x[6] + x[7] + x[8] + T(-1)*x[9] + T(-1)*x[10] +
        T(12)*x[5]^2 + T(-15);
        lcon = T(0), ucon = T(Inf))
    # C(2): lcon=0, G type
    ExaModels.@add_con(c,
        T(-40)*x[5] + T(78)*x[4] + T(-12)*x[3] + T(-62)*x[2] + T(64)*x[1] +
        T(-2)*x[15] + T(2)*x[14] + T(2)*x[12] + T(9)*x[11] + x[7] + T(2)*x[8] + T(-2)*x[9] + T(-1)*x[10] +
        T(24)*x[4]^2 + T(-27);
        lcon = T(0), ucon = T(Inf))
    # C(3): lcon=0, G type
    ExaModels.@add_con(c,
        T(-20)*x[5] + T(-12)*x[4] + T(20)*x[3] + T(-12)*x[2] + T(-20)*x[1] +
        T(-2)*x[13] + T(2)*x[11] + T(4)*x[6] + x[7] + T(3)*x[8] + T(-3)*x[9] + T(-1)*x[10] +
        T(30)*x[3]^2 + T(-36);
        lcon = T(0), ucon = T(Inf))
    # C(4): lcon=0, G type
    ExaModels.@add_con(c,
        T(64)*x[5] + T(-62)*x[4] + T(-12)*x[3] + T(78)*x[2] + T(-40)*x[1] +
        T(-1)*x[15] + T(-4)*x[14] + T(4)*x[12] + T(-1)*x[11] + x[7] + T(2)*x[8] + T(-4)*x[9] + T(-1)*x[10] +
        T(18)*x[2]^2 + T(-18);
        lcon = T(0), ucon = T(Inf))
    # C(5): lcon=0, G type
    ExaModels.@add_con(c,
        T(-20)*x[5] + T(64)*x[4] + T(-20)*x[3] + T(-40)*x[2] + T(60)*x[1] +
        T(-2)*x[14] + x[12] + T(2.8)*x[11] + x[7] + x[8] + T(-5)*x[9] + T(-1)*x[10] +
        T(6)*x[1]^2 + T(-12);
        lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
