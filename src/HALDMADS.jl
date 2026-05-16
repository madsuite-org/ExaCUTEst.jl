# HALDMADS: Hald-Madsen minimax problem in 5 variables (Hald & Madsen 1981)
# 6 variables: X1..X5 (x[1..5]) and U (x[6])
# Objective: minimize U = x[6]
# 42 constraints (21 pairs F(I) and MF(I)):
#   F(I)  (L type, lcon=0): group value = -U + HM(x,Y(I)) - EY(I) <= 0
#          → x[6] - HM(x,Y(I)) + EY(I) >= 0
#   MF(I) (L type, lcon=0): group value = -U - HM(x,Y(I)) + EY(I) <= 0
#          → x[6] + HM(x,Y(I)) - EY(I) >= 0
# where Y(I) = -1.0 + 0.1*(I-1) for I=1..21
#       EY(I) = exp(Y(I))
#
# HM element: F(V1..V5; Y) = (V1 + Y*V2) / (1 + V3*Y + V4*Y^2 + V5*Y^3)
# Start: X1=0.5, rest=0 (XV HALDMADS X1 0.5)
# All variables free
function haldmads_model(; T = Float64, backend = nothing, kwargs...)
    # Precompute Y(I) and EY(I)
    Y_vals = T[-T(1) + T(0.1) * T(i - 1) for i in 1:21]
    EY_vals = T[exp(y) for y in Y_vals]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 6; lvar = T(-Inf), uvar = T(Inf),
                       start = T[0.5, 0, 0, 0, 0, 0])

    # Objective: minimize U = x[6]
    ExaModels.@add_obj(c, x[6])

    ExaModels.@add_par(c, YY, Y_vals)
    ExaModels.@add_par(c, EY, EY_vals)

    # F(I): x[6] - HM(x,Y(I)) + EY(I) >= 0
    ExaModels.@add_con(c,
        x[6] - (x[1] + YY[i] * x[2]) / (T(1) + x[3] * YY[i] + x[4] * YY[i]^2 + x[5] * YY[i]^3) + EY[i]
        for i in 1:21;
        lcon = T(0), ucon = T(Inf))

    # MF(I): x[6] + HM(x,Y(I)) - EY(I) >= 0
    ExaModels.@add_con(c,
        x[6] + (x[1] + YY[i] * x[2]) / (T(1) + x[3] * YY[i] + x[4] * YY[i]^2 + x[5] * YY[i]^3) - EY[i]
        for i in 1:21;
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
