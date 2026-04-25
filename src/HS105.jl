# HS105: Mixture distribution fitting (8 variables, 1 inequality constraint, 235 obs)
# Source: Hock & Schittkowski, problem 105
# SIF: 235 N-type groups OBJ(I) with GROUP TYPE LOG: F(T) = -log(P*T)
#   where T = A(I) + B(I) + C(I)
#   A(I) = (x1/x6) * exp(-0.5*((Y[i]-x3)/x6)^2)
#   B(I) = (x2/x7) * exp(-0.5*((Y[i]-x4)/x7)^2)
#   C(I) = ((1-x1-x2)/x8) * exp(-0.5*((Y[i]-x5)/x8)^2)
#   P = 0.3989422804014327  (= 1/sqrt(2*pi))
#   Constraint C1: -x1 - x2 + 1 >= 0  (G-type, RHS=-1, group value = -x1-x2+1)
const P_HS105 = 0.3989422804014327

const Y_HS105 = let
    y = zeros(235)
    y[1]      = 95.0
    y[2]      = 105.0
    y[3:6]   .= 110.0
    y[7:10]  .= 115.0
    y[11:25] .= 120.0
    y[26:40] .= 125.0
    y[41:55] .= 130.0
    y[56:68] .= 135.0
    y[69:89] .= 140.0
    y[90:101].= 145.0
    y[102:118].= 150.0
    y[119:122].= 155.0
    y[123:142].= 160.0
    y[143:150].= 165.0
    y[151:167].= 170.0
    y[168:175].= 175.0
    y[176:181].= 180.0
    y[182:187].= 185.0
    y[188:194].= 190.0
    y[195:198].= 195.0
    y[199:201].= 200.0
    y[202:204].= 205.0
    y[205:212].= 210.0
    y[213]     = 215.0
    y[214:219].= 220.0
    y[220:224].= 230.0
    y[225]     = 235.0
    y[226:232].= 240.0
    y[233]     = 245.0
    y[234:235].= 250.0
    y
end

function hs105_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 8;
        lvar  = T[0.001, 0.001, 100.0, 130.0, 170.0, 5.0,  5.0,  5.0 ],
        uvar  = T[0.499, 0.499, 180.0, 210.0, 240.0, 25.0, 25.0, 25.0],
        start = T[0.1,   0.2,   100.0, 125.0, 175.0, 11.2, 13.2, 15.8])
    ExaModels.@add_par(c, Y, Y_HS105)
    ExaModels.@add_obj(c,
        -log(T(P_HS105) * (
            (x[1]/x[6]) * exp(T(-0.5) * ((Y[i] - x[3])/x[6])^2) +
            (x[2]/x[7]) * exp(T(-0.5) * ((Y[i] - x[4])/x[7])^2) +
            ((T(1) - x[1] - x[2])/x[8]) * exp(T(-0.5) * ((Y[i] - x[5])/x[8])^2)
        )) for i in 1:235)
    # G-type constraint: -x1 - x2 + 1 >= 0  (lcon=0, ucon=Inf)
    ExaModels.@add_con(c, -x[1] - x[2] + T(1); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
