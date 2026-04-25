# HS67: Colville's problem 8 / H&S problem 67 (3 variables, 14 constraints)
# Source: problem 67 in Hock & Schittkowski (1981); originally Colville (1968) problem 8
# SIF classification: OOI2-AN-3-14
#
# Variables: x[1]=X(1), x[2]=X(2), x[3]=X(3)
#   lvar = [1e-5, 1e-5, 1e-5]
#   uvar = [2000, 16000, 120]
#   start = [1745, 12000, 110]
#
# The SIF uses an external Fortran procedure for the element types, which implements
# an iterative fixed-point computation. This implementation uses the closed-form
# algebraic fixed points (quadratic-formula roots), which match the exact mathematical
# fixed points of the iterative system.
#
# NOTE: CUTEst evaluates the elements by running the Fortran iteration until
# |y2c - y2| < 0.001 (loose tolerance), so CUTEst values may differ slightly from
# the exact algebraic fixed points computed here.
#
# Intermediate quantities (all functions of x[1],x[2],x[3]):
#   y2  = fixed point of y2 = 0.01*x[1]*(112 + 13.167*y6 - 0.6667*y6^2)
#           where y6 = (x[2] + 1.22*y2 - x[1]) / x[1]
#         Closed form: positive root of a*y2^2 + b*y2 + c = 0
#   y3  = 1.22*y2 - x[1]
#   y6  = (x[2] + y3) / x[1]
#   y4  = fixed point of y4 = 98000*x[3] / (y2*y7 + 1000*x[3])
#           where y7 = 35.82 - 0.222*(3*y5-133), y5 = 86.35 + 1.098*y6 - 0.038*y6^2 + 0.325*(y4-89)
#         Closed form: positive root of q4*y4^2 + p4*y4 + r4 = 0
#   y5  = 86.35 + 1.098*y6 - 0.038*y6^2 + 0.325*(y4 - 89)
#   y7  = 27.10095 - 0.731268*y6 + 0.025308*y6^2 - 0.21645*y4
#   y8  = 3*y5 - 133
#
# Objective: 5.04*x1 + 0.035*x2 + 10*x3 - 0.063*y2*y5 + 3.36*y3
#
# Constraints (14 = 7 pairs of G/L):
#   AG(i) G type, ≥ A(i):  lcon=0, ucon=Inf  (residual = elem - A(i))
#   AL(i) L type, ≤ A(8+i): lcon=-Inf, ucon=0 (residual = elem - A(8+i))
#   A = [0,0,85,90,3,0.01,145, 5000,2000,93,95,12,4,162]
#   Constraint ordering (CUTEst): [AG1,AL1, AG2,AL2, ..., AG7,AL7]
#   Elements: E2=y2, E3=y3, E4=y4, E5=y5, E6=y6, E7=y7, E8=y8
#
# Registration of y2 bivariate fixed-point function with exact derivatives
# via implicit function theorem:

const _HS67_K1 = 0.006667 * 1.4884   # coefficient for a(x1) = K1/x1
const _HS67_K2 = 2.0 * 0.006667 * 1.22  # coefficient in b: K2*(x2-x1)/x1
const _HS67_K3 = 0.13167

# y2*(x1,x2): positive root of a*y2^2 + b*y2 + c = 0
function _hs67_y2(x1::Real, x2::Real)
    k1 = _HS67_K1; k2 = _HS67_K2; k3 = _HS67_K3
    dx = x2 - x1
    a = k1 / x1
    b = 0.8393626 + k2 * dx / x1
    c = k1 * dx^2 / x1 - 1.12 * x1 - k3 * dx
    (-b + sqrt(b^2 - 4.0 * a * c)) / (2.0 * a)
end

# Derivatives of y2 via implicit function theorem
# F(y2,x1,x2) = a*y2^2 + b*y2 + c = 0
# dy2/dxi = -(dF/dxi) / (dF/dy2),  dF/dy2 = 2*a*y2 + b

function _dhs67_y2_1(x1::Real, x2::Real)
    k1 = _HS67_K1; k2 = _HS67_K2; k3 = _HS67_K3
    dx = x2 - x1
    a = k1 / x1
    b = 0.8393626 + k2 * dx / x1
    c = k1 * dx^2 / x1 - 1.12 * x1 - k3 * dx
    y2 = (-b + sqrt(b^2 - 4.0*a*c)) / (2.0*a)

    da = -k1 / x1^2                                  # da/dx1
    db = -k2 * x2 / x1^2                             # db/dx1
    dc = k1 * dx * (-x1 - x2) / x1^2 - 1.12 + k3   # dc/dx1

    Fx1 = da * y2^2 + db * y2 + dc
    Fy2 = 2.0 * a * y2 + b
    -Fx1 / Fy2
end

function _dhs67_y2_2(x1::Real, x2::Real)
    k1 = _HS67_K1; k2 = _HS67_K2; k3 = _HS67_K3
    dx = x2 - x1
    a = k1 / x1
    b = 0.8393626 + k2 * dx / x1
    c = k1 * dx^2 / x1 - 1.12 * x1 - k3 * dx
    y2 = (-b + sqrt(b^2 - 4.0*a*c)) / (2.0*a)

    db = k2 / x1                       # db/dx2
    dc = 2.0 * k1 * dx / x1 - k3      # dc/dx2

    Fx2 = db * y2 + dc
    Fy2 = 2.0 * a * y2 + b
    -Fx2 / Fy2
end

# Second-order derivatives of y2 via repeated implicit differentiation
# d²y2/dx_i dx_j = -(d/dxj[F_xi] + d/dxj[Fy2] * dy2/dxi) / Fy2
# where d/dxj[F_xi] = F_xixj + F_xiy2 * dy2/dxj (treating all as total derivatives)
# and d/dxj[Fy2] = Fy2xj + Fy2y2 * dy2/dxj = Fy2xj + 2a * dy2/dxj

function _d2hs67_y2_11(x1::Real, x2::Real)
    k1 = _HS67_K1; k2 = _HS67_K2; k3 = _HS67_K3
    dx = x2 - x1
    a = k1 / x1
    b = 0.8393626 + k2 * dx / x1
    c = k1 * dx^2 / x1 - 1.12 * x1 - k3 * dx
    y2 = (-b + sqrt(b^2 - 4.0*a*c)) / (2.0*a)

    da  = -k1 / x1^2
    db1 = -k2 * x2 / x1^2
    dc1 = k1 * dx * (-x1 - x2) / x1^2 - 1.12 + k3

    Fy2 = 2.0 * a * y2 + b
    dy2_1 = -(da * y2^2 + db1 * y2 + dc1) / Fy2

    # Second partial derivatives (w.r.t. x1, treating y2 as fixed)
    d2a   = 2.0 * k1 / x1^3
    d2b11 = 2.0 * k2 * x2 / x1^3    # d²b/dx1² = d/dx1[-k2*x2/x1^2] = 2k2*x2/x1^3
    d2c11 = 2.0 * k1 * x2^2 / x1^3  # d²c/dx1² = d/dx1[k1*(x1^2-x2^2)/x1^2] = 2k1*x2^2/x1^3

    # Fx1x1 (pure partial, y2 fixed) + cross terms from chain rule:
    Fx1x1 = d2a * y2^2 + d2b11 * y2 + d2c11
    Fx1y2 = 2.0 * da * y2 + db1   # = Fy2x1
    Fy2x1 = Fx1y2

    -(Fx1x1 + 2.0 * Fx1y2 * dy2_1 + 2.0 * a * dy2_1^2) / Fy2
end

function _d2hs67_y2_12(x1::Real, x2::Real)
    k1 = _HS67_K1; k2 = _HS67_K2; k3 = _HS67_K3
    dx = x2 - x1
    a = k1 / x1
    b = 0.8393626 + k2 * dx / x1
    c = k1 * dx^2 / x1 - 1.12 * x1 - k3 * dx
    y2 = (-b + sqrt(b^2 - 4.0*a*c)) / (2.0*a)

    da  = -k1 / x1^2
    db1 = -k2 * x2 / x1^2
    dc1 = k1 * dx * (-x1 - x2) / x1^2 - 1.12 + k3
    db2 = k2 / x1
    dc2 = 2.0 * k1 * dx / x1 - k3

    Fy2  = 2.0 * a * y2 + b
    dy2_1 = -(da * y2^2 + db1 * y2 + dc1) / Fy2
    dy2_2 = -(db2 * y2 + dc2) / Fy2

    # d²b/dx1dx2 = d/dx2[-k2*x2/x1^2] = -k2/x1^2
    # d²c/dx1dx2: c has term k1*(x1^2-x2^2)/x1^2 = k1*(1-(x2/x1)^2)
    #   dc/dx1 = k1*(-2x2^2/x1^3)*... actually: dc/dx1 = k1*(x1^2-x2^2)/x1^2 - 0.98833
    #   Hmm wait: dc/dx1 was computed as k1*dx*(-x1-x2)/x1^2 - 1.12 + k3
    #   d/dx2[k1*dx*(-x1-x2)/x1^2] = k1*((-x1-x2)/x1^2 + dx*(-1)/x1^2)
    #                                = k1*(-x1-x2-dx)/x1^2
    #                                = k1*(-x1-x2-x2+x1)/x1^2 = k1*(-2x2)/x1^2
    d2b12 = -k2 / x1^2
    d2c12 = -2.0 * k1 * x2 / x1^2

    Fx1x2 = d2b12 * y2 + d2c12
    Fx1y2 = 2.0 * da * y2 + db1
    Fy2x2 = db2   # d/dx2[Fy2] = d/dx2[2ay2+b] = db/dx2 (a has no x2 dep)

    -(Fx1x2 + Fx1y2 * dy2_2 + (Fy2x2 + 2.0 * a * dy2_2) * dy2_1) / Fy2
end

function _d2hs67_y2_22(x1::Real, x2::Real)
    k1 = _HS67_K1; k2 = _HS67_K2; k3 = _HS67_K3
    dx = x2 - x1
    a = k1 / x1
    b = 0.8393626 + k2 * dx / x1
    c = k1 * dx^2 / x1 - 1.12 * x1 - k3 * dx
    y2 = (-b + sqrt(b^2 - 4.0*a*c)) / (2.0*a)

    db2 = k2 / x1
    dc2 = 2.0 * k1 * dx / x1 - k3

    Fy2  = 2.0 * a * y2 + b
    dy2_2 = -(db2 * y2 + dc2) / Fy2

    # d²b/dx2² = 0,  d²c/dx2² = 2*k1/x1
    d2c22 = 2.0 * k1 / x1
    Fx2x2 = d2c22
    Fy2x2 = db2

    -(Fx2x2 + (Fy2x2 + 2.0 * a * dy2_2) * dy2_2) / Fy2
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
    #   A(1..7) = [0, 0, 85, 90, 3, 0.01, 145]  for G-type (≥)
    #   A(8..14)= [5000, 2000, 93, 95, 12, 4, 162]  for L-type (≤)
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
        # y4 fixed point: positive root of -0.21645*y2*y4^2 + (y2*K1+1000*x3)*y4 - 98000*x3 = 0
        K1v = T(27.10095) - T(0.731268) * y6 + T(0.025308) * y6^2
        a4 = -T(0.21645) * y2
        b4 = y2 * K1v + T(1000) * x3
        c4 = -T(98000) * x3
        y4 = (-b4 + sqrt(b4^2 - T(4) * a4 * c4)) / (T(2) * a4)
        y5 = T(86.35) + T(1.098) * y6 - T(0.038) * y6^2 + T(0.325) * (y4 - T(89))
        T(5.04) * x1 + T(0.035) * x2 + T(10) * x3 - T(0.063) * y2 * y5 + T(3.36) * y3
    end)

    # Constraints: 14 = 7 pairs of (AG_i, AL_i), interleaved
    # Element ordering: E2=y2, E3=y3, E4=y4, E5=y5, E6=y6, E7=y7, E8=y8
    # AG(i): elem_i - AG_rhs[i] >= 0  →  lcon=0, ucon=Inf
    # AL(i): elem_i - AL_rhs[i] <= 0  →  lcon=-Inf, ucon=0
    # CUTEst ordering: [AG1,AL1, AG2,AL2, AG3,AL3, AG4,AL4, AG5,AL5, AG6,AL6, AG7,AL7]

    # Pair 1: element = y2 (E2)
    ExaModels.@add_con(c, _hs67_y2(x[1], x[2]) - AG_rhs[1]; lcon = T(0),    ucon = T(Inf))
    ExaModels.@add_con(c, _hs67_y2(x[1], x[2]) - AL_rhs[1]; lcon = T(-Inf), ucon = T(0))

    # Pair 2: element = y3 = 1.22*y2 - x1
    ExaModels.@add_con(c, T(1.22) * _hs67_y2(x[1], x[2]) - x[1] - AG_rhs[2];
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, T(1.22) * _hs67_y2(x[1], x[2]) - x[1] - AL_rhs[2];
                       lcon = T(-Inf), ucon = T(0))

    # Pairs 3–7: elements y4, y5, y6, y7, y8 depend on y2, y6, y4 (require full expression)
    # These are computed inline as a macro helper
    for (i, (ag, al)) in enumerate(zip(AG_rhs[3:7], AL_rhs[3:7]))
        ag_val = T(ag); al_val = T(al)
        if i == 1  # y4
            ExaModels.@add_con(c, let
                x1=x[1]; x2=x[2]; x3=x[3]
                y2 = _hs67_y2(x1, x2)
                y3 = T(1.22)*y2 - x1
                y6 = (x2+y3)/x1
                K1v = T(27.10095)-T(0.731268)*y6+T(0.025308)*y6^2
                a4=-T(0.21645)*y2; b4=y2*K1v+T(1000)*x3; c4=-T(98000)*x3
                (-b4+sqrt(b4^2-T(4)*a4*c4))/(T(2)*a4) - ag_val
            end; lcon=T(0), ucon=T(Inf))
            ExaModels.@add_con(c, let
                x1=x[1]; x2=x[2]; x3=x[3]
                y2 = _hs67_y2(x1, x2)
                y3 = T(1.22)*y2 - x1
                y6 = (x2+y3)/x1
                K1v = T(27.10095)-T(0.731268)*y6+T(0.025308)*y6^2
                a4=-T(0.21645)*y2; b4=y2*K1v+T(1000)*x3; c4=-T(98000)*x3
                (-b4+sqrt(b4^2-T(4)*a4*c4))/(T(2)*a4) - al_val
            end; lcon=T(-Inf), ucon=T(0))
        elseif i == 2  # y5
            ExaModels.@add_con(c, let
                x1=x[1]; x2=x[2]; x3=x[3]
                y2=_hs67_y2(x1,x2); y3=T(1.22)*y2-x1; y6=(x2+y3)/x1
                K1v=T(27.10095)-T(0.731268)*y6+T(0.025308)*y6^2
                a4=-T(0.21645)*y2; b4=y2*K1v+T(1000)*x3
                y4=(-b4+sqrt(b4^2-T(4)*a4*(-T(98000)*x3)))/(T(2)*a4)
                T(86.35)+T(1.098)*y6-T(0.038)*y6^2+T(0.325)*(y4-T(89)) - ag_val
            end; lcon=T(0), ucon=T(Inf))
            ExaModels.@add_con(c, let
                x1=x[1]; x2=x[2]; x3=x[3]
                y2=_hs67_y2(x1,x2); y3=T(1.22)*y2-x1; y6=(x2+y3)/x1
                K1v=T(27.10095)-T(0.731268)*y6+T(0.025308)*y6^2
                a4=-T(0.21645)*y2; b4=y2*K1v+T(1000)*x3
                y4=(-b4+sqrt(b4^2-T(4)*a4*(-T(98000)*x3)))/(T(2)*a4)
                T(86.35)+T(1.098)*y6-T(0.038)*y6^2+T(0.325)*(y4-T(89)) - al_val
            end; lcon=T(-Inf), ucon=T(0))
        elseif i == 3  # y6
            ExaModels.@add_con(c, let
                x1=x[1]; x2=x[2]
                y2=_hs67_y2(x1,x2); y3=T(1.22)*y2-x1
                (x2+y3)/x1 - ag_val
            end; lcon=T(0), ucon=T(Inf))
            ExaModels.@add_con(c, let
                x1=x[1]; x2=x[2]
                y2=_hs67_y2(x1,x2); y3=T(1.22)*y2-x1
                (x2+y3)/x1 - al_val
            end; lcon=T(-Inf), ucon=T(0))
        elseif i == 4  # y7 = 27.10095 - 0.731268*y6 + 0.025308*y6^2 - 0.21645*y4
            ExaModels.@add_con(c, let
                x1=x[1]; x2=x[2]; x3=x[3]
                y2=_hs67_y2(x1,x2); y3=T(1.22)*y2-x1; y6=(x2+y3)/x1
                K1v=T(27.10095)-T(0.731268)*y6+T(0.025308)*y6^2
                a4=-T(0.21645)*y2; b4=y2*K1v+T(1000)*x3
                y4=(-b4+sqrt(b4^2-T(4)*a4*(-T(98000)*x3)))/(T(2)*a4)
                K1v - T(0.21645)*y4 - ag_val
            end; lcon=T(0), ucon=T(Inf))
            ExaModels.@add_con(c, let
                x1=x[1]; x2=x[2]; x3=x[3]
                y2=_hs67_y2(x1,x2); y3=T(1.22)*y2-x1; y6=(x2+y3)/x1
                K1v=T(27.10095)-T(0.731268)*y6+T(0.025308)*y6^2
                a4=-T(0.21645)*y2; b4=y2*K1v+T(1000)*x3
                y4=(-b4+sqrt(b4^2-T(4)*a4*(-T(98000)*x3)))/(T(2)*a4)
                K1v - T(0.21645)*y4 - al_val
            end; lcon=T(-Inf), ucon=T(0))
        else  # i == 5, y8 = 3*y5 - 133
            ExaModels.@add_con(c, let
                x1=x[1]; x2=x[2]; x3=x[3]
                y2=_hs67_y2(x1,x2); y3=T(1.22)*y2-x1; y6=(x2+y3)/x1
                K1v=T(27.10095)-T(0.731268)*y6+T(0.025308)*y6^2
                a4=-T(0.21645)*y2; b4=y2*K1v+T(1000)*x3
                y4=(-b4+sqrt(b4^2-T(4)*a4*(-T(98000)*x3)))/(T(2)*a4)
                y5=T(86.35)+T(1.098)*y6-T(0.038)*y6^2+T(0.325)*(y4-T(89))
                T(3)*y5 - T(133) - ag_val
            end; lcon=T(0), ucon=T(Inf))
            ExaModels.@add_con(c, let
                x1=x[1]; x2=x[2]; x3=x[3]
                y2=_hs67_y2(x1,x2); y3=T(1.22)*y2-x1; y6=(x2+y3)/x1
                K1v=T(27.10095)-T(0.731268)*y6+T(0.025308)*y6^2
                a4=-T(0.21645)*y2; b4=y2*K1v+T(1000)*x3
                y4=(-b4+sqrt(b4^2-T(4)*a4*(-T(98000)*x3)))/(T(2)*a4)
                y5=T(86.35)+T(1.098)*y6-T(0.038)*y6^2+T(0.325)*(y4-T(89))
                T(3)*y5 - T(133) - al_val
            end; lcon=T(-Inf), ucon=T(0))
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end
