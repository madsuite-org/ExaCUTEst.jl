# HS268: Quadratic programming problem (5 variables, 5 inequality constraints)
# Source: Schittkowski (1987), "More Test Examples for Nonlinear Programming Codes"
# SIF: XN NONL (identity, no scale): sum_{i,j} D(i,j)*x_i*x_j - CONST_NONL = sum D*x*x + 14463
#      XN LINEAR (identity, scale=-0.5): contribution = v/sigma = sum B(i)*x_i / (-0.5) = -2*B.x
#      XG C1..C5: G-type inequality constraints (>=0); CONSTANTS provide rhs
# Obj = sum_{i,j} D(i,j)*x_i*x_j + 14463 - 2*sum_i B(i)*x_i
function hs268_model(; T = Float64, backend = nothing, kwargs...)
    # D matrix (5x5, symmetric), stored row-major: DD[(i-1)*5+j] = D[i,j]
    DD_data = T[
        10197.0, -12454.0, -1013.0, 1948.0, 329.0,
        -12454.0, 20909.0, -1733.0, -4914.0, -186.0,
        -1013.0, -1733.0, 1755.0, 1089.0, -174.0,
        1948.0, -4914.0, 1089.0, 1515.0, -22.0,
        329.0, -186.0, -174.0, -22.0, 27.0,
    ]
    BB_data = T[-9170.0, 17099.0, -2271.0, -4336.0, -43.0]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = fill(T(-Inf), 5), uvar = fill(T(Inf), 5),
                       start = fill(T(1.0), 5))
    ExaModels.@add_par(c, DD, DD_data)
    ExaModels.@add_par(c, BB, BB_data)
    # NONL group contribution: sum_{i,j} D(i,j)*x_i*x_j + 14463
    ExaModels.@add_obj(c, DD[(i-1)*5+j] * x[i] * x[j] for i in 1:5, j in 1:5)
    # constant offset from CONSTANT_NONL = -14463: contribution += 14463
    ExaModels.@add_obj(c, T(14463) + T(0) * x[k] for k in 1:1)
    # LINEAR group contribution: -2 * sum_i B(i)*x_i  (v/sigma with sigma=-0.5)
    ExaModels.@add_obj(c, T(-2) * BB[i] * x[i] for i in 1:5)
    # Constraints C1..C5 (XG = G type, lcon=0 ucon=Inf)
    # CUTEst residual form: g(x) - rhs
    # C1: -(x1+x2+x3+x4+x5) - (-5) = -(sum x) + 5
    ExaModels.@add_con(c, -(x[1]+x[2]+x[3]+x[4]+x[5]) - T(-5); lcon = T(0), ucon = T(Inf))
    # C2: 10x1+10x2-3x3+5x4+4x5 - 20
    ExaModels.@add_con(c, T(10)*x[1]+T(10)*x[2]-T(3)*x[3]+T(5)*x[4]+T(4)*x[5] - T(20); lcon = T(0), ucon = T(Inf))
    # C3: -8x1+x2-2x3-5x4+3x5 - (-40)
    ExaModels.@add_con(c, -T(8)*x[1]+x[2]-T(2)*x[3]-T(5)*x[4]+T(3)*x[5] - T(-40); lcon = T(0), ucon = T(Inf))
    # C4: 8x1-x2+2x3+5x4-3x5 - 11
    ExaModels.@add_con(c, T(8)*x[1]-x[2]+T(2)*x[3]+T(5)*x[4]-T(3)*x[5] - T(11); lcon = T(0), ucon = T(Inf))
    # C5: -4x1-2x2+3x3-5x4+x5 - (-30)
    ExaModels.@add_con(c, -T(4)*x[1]-T(2)*x[2]+T(3)*x[3]-T(5)*x[4]+x[5] - T(-30); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
