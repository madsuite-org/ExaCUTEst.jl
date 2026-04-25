# HS67: Colville's problem 8 / H&S problem 67 (3 variables, 14 constraints)
# Source: problem 67 in Hock & Schittkowski (1981); originally Colville (1968) problem 8
# SIF classification: OOI2-AN-3-14
#
# Variables: x[1]=X(1), x[2]=X(2), x[3]=X(3)
#   lvar = [1e-5, 1e-5, 1e-5]
#   uvar = [2000, 16000, 120]
#   start = [1745, 12000, 110]
#
# The SIF uses an external Fortran procedure that computes y2, y4 via fixed-point
# iteration, propagating exact gradients/Hessians through each iteration step.
# This implementation uses closed-form algebraic fixed points (quadratic formula),
# which reproduce the same values as the Fortran iteration at convergence.
#
# Intermediate quantities (all functions of x[1]=x1, x[2]=x2, x[3]=x3):
#   y2: positive root of KA/x1*y2^2 + (KB + KC*(x2-x1)/x1)*y2
#                        - (KD*x1 + KE*(x2-x1) - KF*(x2-x1)^2/x1) = 0
#     where KA=0.6667*1.22^2, KB=100-13.167*1.22, KC=2*0.6667*1.22,
#           KD=112, KE=13.167, KF=0.6667
#   y3  = 1.22*y2 - x1
#   y6  = (x2 + y3) / x1  (= (x2 + 1.22*y2 - x1)/x1)
#   y4: positive root of -0.21645*y2*y4^2 + (y2*K + 1000*x3)*y4 - 98000*x3 = 0
#     where K = 27.10095 - 0.731268*y6 + 0.025308*y6^2
#   y5  = 86.35 + 1.098*y6 - 0.038*y6^2 + 0.325*(y4 - 89)
#   y7  = 27.10095 - 0.731268*y6 + 0.025308*y6^2 - 0.21645*y4
#   y8  = 3*y5 - 133
#
# Objective: 5.04*x1 + 0.035*x2 + 10*x3 - 0.063*y2*y5 + 3.36*y3
#
# Constraints (14 = 7 pairs of G/L), with CUTEst ordering [AG1,AL1,...,AG7,AL7]:
#   A_G = [0, 0, 85, 90, 3, 0.01, 145]   (G-type: element >= A_G[i])
#   A_L = [5000, 2000, 93, 95, 12, 4, 162] (L-type: element <= A_L[i])
#   Elements: E2=y2, E3=y3, E4=y4, E5=y5, E6=y6, E7=y7, E8=y8

# Constants for y2 quadratic
const _HS67_KA = 0.6667 * 1.22 * 1.22     # = 0.6667 * 1.4884
const _HS67_KB = 100.0 - 13.167 * 1.22    # = 83.93626
const _HS67_KC = 2.0 * 0.6667 * 1.22      # = 1.62675...
const _HS67_KD = 112.0
const _HS67_KE = 13.167
const _HS67_KF = 0.6667

# y2(x1,x2): positive root of the fixed-point quadratic
function _hs67_y2(x1::Real, x2::Real)
    dx = x2 - x1
    a = _HS67_KA / x1
    b = _HS67_KB + _HS67_KC * dx / x1
    c = -(_HS67_KD * x1 + _HS67_KE * dx - _HS67_KF * dx * dx / x1)
    (-b + sqrt(b * b - 4 * a * c)) / (2 * a)
end

# First derivatives of y2 via implicit function theorem
# F(y2,x1,x2) = a(x1)*y2^2 + b(x1,x2)*y2 + c(x1,x2) = 0
# dy2/dxi = -F_{xi} / F_{y2},  where F_{y2} = 2*a*y2 + b

function _dhs67_y2_1(x1::Real, x2::Real)
    dx = x2 - x1
    a  = _HS67_KA / x1
    b  = _HS67_KB + _HS67_KC * dx / x1
    c  = -(_HS67_KD * x1 + _HS67_KE * dx - _HS67_KF * dx * dx / x1)
    y2 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)

    da1 = -_HS67_KA / x1^2
    db1 = -_HS67_KC * x2 / x1^2
    dc1 = -(_HS67_KD - _HS67_KE - _HS67_KF * dx * (-x1 - x2) / x1^2)

    Fx1 = da1 * y2 * y2 + db1 * y2 + dc1
    Fy2 = 2 * a * y2 + b
    -Fx1 / Fy2
end

function _dhs67_y2_2(x1::Real, x2::Real)
    dx = x2 - x1
    a  = _HS67_KA / x1
    b  = _HS67_KB + _HS67_KC * dx / x1
    c  = -(_HS67_KD * x1 + _HS67_KE * dx - _HS67_KF * dx * dx / x1)
    y2 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)

    db2 = _HS67_KC / x1
    dc2 = -(_HS67_KE - _HS67_KF * 2 * dx / x1)

    Fx2 = db2 * y2 + dc2
    Fy2 = 2 * a * y2 + b
    -Fx2 / Fy2
end

# Second derivatives via twice-differentiated IFT
# d²y2/dxi dxj = -(F_{xixj} + F_{y2xi}*dy2/dxj + F_{y2xj}*dy2/dxi + F_{y2y2}*dy2/dxi*dy2/dxj) / Fy2

function _d2hs67_y2_11(x1::Real, x2::Real)
    dx = x2 - x1
    a  = _HS67_KA / x1
    b  = _HS67_KB + _HS67_KC * dx / x1
    c  = -(_HS67_KD * x1 + _HS67_KE * dx - _HS67_KF * dx * dx / x1)
    y2 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)

    da1 = -_HS67_KA / x1^2
    db1 = -_HS67_KC * x2 / x1^2
    dc1 = -(_HS67_KD - _HS67_KE - _HS67_KF * dx * (-x1 - x2) / x1^2)

    Fy2  = 2 * a * y2 + b
    Fx1  = da1 * y2 * y2 + db1 * y2 + dc1
    dy2_1 = -Fx1 / Fy2

    # Second partials of F w.r.t. x1
    d2a11 = 2 * _HS67_KA / x1^3
    d2b11 = 2 * _HS67_KC * x2 / x1^3
    d2c11 = 2 * _HS67_KF * x2 * x2 / x1^3   # = 2*KF*x2^2/x1^3

    Fy2x1  = 2 * da1 * y2 + db1   # d(Fy2)/dx1 treating y2 fixed
    Fy2y2  = 2 * a
    Fx1x1  = d2a11 * y2 * y2 + d2b11 * y2 + d2c11

    -(Fx1x1 + 2 * Fy2x1 * dy2_1 + Fy2y2 * dy2_1 * dy2_1) / Fy2
end

function _d2hs67_y2_12(x1::Real, x2::Real)
    dx = x2 - x1
    a  = _HS67_KA / x1
    b  = _HS67_KB + _HS67_KC * dx / x1
    c  = -(_HS67_KD * x1 + _HS67_KE * dx - _HS67_KF * dx * dx / x1)
    y2 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)

    da1 = -_HS67_KA / x1^2
    db1 = -_HS67_KC * x2 / x1^2
    dc1 = -(_HS67_KD - _HS67_KE - _HS67_KF * dx * (-x1 - x2) / x1^2)
    db2 = _HS67_KC / x1
    dc2 = -(_HS67_KE - _HS67_KF * 2 * dx / x1)

    Fy2  = 2 * a * y2 + b
    Fx1  = da1 * y2 * y2 + db1 * y2 + dc1
    Fx2  = db2 * y2 + dc2
    dy2_1 = -Fx1 / Fy2
    dy2_2 = -Fx2 / Fy2

    # d2b/dx1dx2 = -KC/x1^2,  d2c/dx1dx2 = -2*KF*x2/x1^2
    d2b12 = -_HS67_KC / x1^2
    d2c12 = -2 * _HS67_KF * x2 / x1^2

    Fy2x1 = 2 * da1 * y2 + db1
    Fy2x2 = db2
    Fy2y2 = 2 * a
    Fx1x2 = d2b12 * y2 + d2c12

    -(Fx1x2 + Fy2x1 * dy2_2 + Fy2x2 * dy2_1 + Fy2y2 * dy2_1 * dy2_2) / Fy2
end

function _d2hs67_y2_22(x1::Real, x2::Real)
    dx = x2 - x1
    a  = _HS67_KA / x1
    b  = _HS67_KB + _HS67_KC * dx / x1
    c  = -(_HS67_KD * x1 + _HS67_KE * dx - _HS67_KF * dx * dx / x1)
    y2 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)

    db2 = _HS67_KC / x1
    dc2 = -(_HS67_KE - _HS67_KF * 2 * dx / x1)

    Fy2  = 2 * a * y2 + b
    Fx2  = db2 * y2 + dc2
    dy2_2 = -Fx2 / Fy2

    # d2c/dx2^2 = 2*KF/x1
    d2c22 = 2 * _HS67_KF / x1

    Fy2x2 = db2
    Fy2y2 = 2 * a
    Fx2x2 = d2c22

    -(Fx2x2 + 2 * Fy2x2 * dy2_2 + Fy2y2 * dy2_2 * dy2_2) / Fy2
end

ExaModels.@register_bivariate(
    _hs67_y2,
    _dhs67_y2_1,
    _dhs67_y2_2,
    _d2hs67_y2_11,
    _d2hs67_y2_12,
    _d2hs67_y2_22,
)

function hs67_model(; T = Float64, backend = nothing, kwargs...)
    # Constraint RHS values from SIF:
    #   A(1..7)  for G-type (>=): element - A(i) >= 0
    #   A(8..14) for L-type (<=): element - A(i+7) <= 0
    AG_rhs = T[0.0, 0.0, 85.0, 90.0, 3.0, 0.01, 145.0]
    AL_rhs = T[5000.0, 2000.0, 93.0, 95.0, 12.0, 4.0, 162.0]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar  = T[1e-5, 1e-5, 1e-5],
        uvar  = T[2000.0, 16000.0, 120.0],
        start = T[1745.0, 12000.0, 110.0])

    # Objective: 5.04*x1 + 0.035*x2 + 10*x3 - 0.063*y2*y5 + 3.36*y3
    ExaModels.@add_obj(c, let
        x1 = x[1]; x2 = x[2]; x3 = x[3]
        y2 = _hs67_y2(x1, x2)
        y3 = T(1.22) * y2 - x1
        y6 = (x2 + y3) / x1
        K  = T(27.10095) - T(0.731268) * y6 + T(0.025308) * y6 * y6
        a4 = -T(0.21645) * y2
        b4 = y2 * K + T(1000) * x3
        y4 = (-b4 + sqrt(b4 * b4 - T(4) * a4 * (-T(98000) * x3))) / (T(2) * a4)
        y5 = T(86.35) + T(1.098) * y6 - T(0.038) * y6 * y6 + T(0.325) * (y4 - T(89))
        T(5.04) * x1 + T(0.035) * x2 + T(10) * x3 - T(0.063) * y2 * y5 + T(3.36) * y3
    end)

    # Pair 1: element = y2
    ExaModels.@add_con(c, _hs67_y2(x[1], x[2]) - AG_rhs[1]; lcon = T(0),    ucon = T(Inf))
    ExaModels.@add_con(c, _hs67_y2(x[1], x[2]) - AL_rhs[1]; lcon = T(-Inf), ucon = T(0))

    # Pair 2: element = y3 = 1.22*y2 - x1
    ExaModels.@add_con(c, T(1.22) * _hs67_y2(x[1], x[2]) - x[1] - AG_rhs[2];
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, T(1.22) * _hs67_y2(x[1], x[2]) - x[1] - AL_rhs[2];
                       lcon = T(-Inf), ucon = T(0))

    # Pair 3: element = y4
    ExaModels.@add_con(c, let
        x1 = x[1]; x2 = x[2]; x3 = x[3]
        y2 = _hs67_y2(x1, x2)
        y3 = T(1.22) * y2 - x1
        y6 = (x2 + y3) / x1
        K  = T(27.10095) - T(0.731268) * y6 + T(0.025308) * y6 * y6
        a4 = -T(0.21645) * y2
        b4 = y2 * K + T(1000) * x3
        (-b4 + sqrt(b4 * b4 - T(4) * a4 * (-T(98000) * x3))) / (T(2) * a4) - AG_rhs[3]
    end; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, let
        x1 = x[1]; x2 = x[2]; x3 = x[3]
        y2 = _hs67_y2(x1, x2)
        y3 = T(1.22) * y2 - x1
        y6 = (x2 + y3) / x1
        K  = T(27.10095) - T(0.731268) * y6 + T(0.025308) * y6 * y6
        a4 = -T(0.21645) * y2
        b4 = y2 * K + T(1000) * x3
        (-b4 + sqrt(b4 * b4 - T(4) * a4 * (-T(98000) * x3))) / (T(2) * a4) - AL_rhs[3]
    end; lcon = T(-Inf), ucon = T(0))

    # Pair 4: element = y5 = 86.35 + 1.098*y6 - 0.038*y6^2 + 0.325*(y4-89)
    ExaModels.@add_con(c, let
        x1 = x[1]; x2 = x[2]; x3 = x[3]
        y2 = _hs67_y2(x1, x2)
        y3 = T(1.22) * y2 - x1
        y6 = (x2 + y3) / x1
        K  = T(27.10095) - T(0.731268) * y6 + T(0.025308) * y6 * y6
        a4 = -T(0.21645) * y2
        b4 = y2 * K + T(1000) * x3
        y4 = (-b4 + sqrt(b4 * b4 - T(4) * a4 * (-T(98000) * x3))) / (T(2) * a4)
        T(86.35) + T(1.098) * y6 - T(0.038) * y6 * y6 + T(0.325) * (y4 - T(89)) - AG_rhs[4]
    end; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, let
        x1 = x[1]; x2 = x[2]; x3 = x[3]
        y2 = _hs67_y2(x1, x2)
        y3 = T(1.22) * y2 - x1
        y6 = (x2 + y3) / x1
        K  = T(27.10095) - T(0.731268) * y6 + T(0.025308) * y6 * y6
        a4 = -T(0.21645) * y2
        b4 = y2 * K + T(1000) * x3
        y4 = (-b4 + sqrt(b4 * b4 - T(4) * a4 * (-T(98000) * x3))) / (T(2) * a4)
        T(86.35) + T(1.098) * y6 - T(0.038) * y6 * y6 + T(0.325) * (y4 - T(89)) - AL_rhs[4]
    end; lcon = T(-Inf), ucon = T(0))

    # Pair 5: element = y6 = (x2 + 1.22*y2 - x1)/x1
    ExaModels.@add_con(c, let
        x1 = x[1]; x2 = x[2]
        y2 = _hs67_y2(x1, x2)
        y3 = T(1.22) * y2 - x1
        (x2 + y3) / x1 - AG_rhs[5]
    end; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, let
        x1 = x[1]; x2 = x[2]
        y2 = _hs67_y2(x1, x2)
        y3 = T(1.22) * y2 - x1
        (x2 + y3) / x1 - AL_rhs[5]
    end; lcon = T(-Inf), ucon = T(0))

    # Pair 6: element = y7 = K - 0.21645*y4  (where K = 27.10095 - 0.731268*y6 + 0.025308*y6^2)
    ExaModels.@add_con(c, let
        x1 = x[1]; x2 = x[2]; x3 = x[3]
        y2 = _hs67_y2(x1, x2)
        y3 = T(1.22) * y2 - x1
        y6 = (x2 + y3) / x1
        K  = T(27.10095) - T(0.731268) * y6 + T(0.025308) * y6 * y6
        a4 = -T(0.21645) * y2
        b4 = y2 * K + T(1000) * x3
        y4 = (-b4 + sqrt(b4 * b4 - T(4) * a4 * (-T(98000) * x3))) / (T(2) * a4)
        K - T(0.21645) * y4 - AG_rhs[6]
    end; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, let
        x1 = x[1]; x2 = x[2]; x3 = x[3]
        y2 = _hs67_y2(x1, x2)
        y3 = T(1.22) * y2 - x1
        y6 = (x2 + y3) / x1
        K  = T(27.10095) - T(0.731268) * y6 + T(0.025308) * y6 * y6
        a4 = -T(0.21645) * y2
        b4 = y2 * K + T(1000) * x3
        y4 = (-b4 + sqrt(b4 * b4 - T(4) * a4 * (-T(98000) * x3))) / (T(2) * a4)
        K - T(0.21645) * y4 - AL_rhs[6]
    end; lcon = T(-Inf), ucon = T(0))

    # Pair 7: element = y8 = 3*y5 - 133
    ExaModels.@add_con(c, let
        x1 = x[1]; x2 = x[2]; x3 = x[3]
        y2 = _hs67_y2(x1, x2)
        y3 = T(1.22) * y2 - x1
        y6 = (x2 + y3) / x1
        K  = T(27.10095) - T(0.731268) * y6 + T(0.025308) * y6 * y6
        a4 = -T(0.21645) * y2
        b4 = y2 * K + T(1000) * x3
        y4 = (-b4 + sqrt(b4 * b4 - T(4) * a4 * (-T(98000) * x3))) / (T(2) * a4)
        y5 = T(86.35) + T(1.098) * y6 - T(0.038) * y6 * y6 + T(0.325) * (y4 - T(89))
        T(3) * y5 - T(133) - AG_rhs[7]
    end; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, let
        x1 = x[1]; x2 = x[2]; x3 = x[3]
        y2 = _hs67_y2(x1, x2)
        y3 = T(1.22) * y2 - x1
        y6 = (x2 + y3) / x1
        K  = T(27.10095) - T(0.731268) * y6 + T(0.025308) * y6 * y6
        a4 = -T(0.21645) * y2
        b4 = y2 * K + T(1000) * x3
        y4 = (-b4 + sqrt(b4 * b4 - T(4) * a4 * (-T(98000) * x3))) / (T(2) * a4)
        y5 = T(86.35) + T(1.098) * y6 - T(0.038) * y6 * y6 + T(0.325) * (y4 - T(89))
        T(3) * y5 - T(133) - AL_rhs[7]
    end; lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
