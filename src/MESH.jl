# MESH: Finite element mesh optimization (maximize minimum triangle angle)
# 41 variables, 48 constraints.
# Source: Prof. Dr. Michael Kraetzschmar, Fachhochschule Flensburg.
# SIF input: M. Kraetzschmar. Classification OOR2-AY-41-48.
#
# CUTEst variable ordering (empirically determined):
#   pos 1..4:   delta4, gamma4, beta4, alpha4  [0,pi]
#   pos 5:      deltamin                        [0,pi]
#   pos 6..9:   delta3, gamma3, beta3, alpha3   [0,pi]
#   pos 10:     f1                              [0,Inf]
#   pos 11..14: delta2, gamma2, beta2, alpha2   [0,pi]
#   pos 15:     f3                              [0,Inf]
#   pos 16..19: delta1, gamma1, beta1, alpha1   [0,pi]
#   pos 20..27: l8, l7, l6, l5, l4, l3, l2, l1 [0,Inf]
#   pos 28..29: y5, x5                          [-Inf,Inf]
#   pos 30..37: y1, y2, y3, x3, x4, x1, x2, y4 (fixed)
#   pos 38..41: f4, fmax, f2, fmin              [0,Inf]
#
# CUTEst constraint ordering:
#   1..4:   wisum(1..4)                equality (linear, no elements)
#   5..8:   alphd(1..4)                inequality [0,Inf]
#   9..12:  betad(1..4)                inequality
#   13..16: gammd(1..4)                inequality
#   17..20: deltd(1..4)                inequality
#   21..24: fmind(1..4)                inequality
#   25..28: fmaxd(1..4)                inequality
#   29..36: seit(1..8)                 equality (nonlinear elements)
#   37..40: skal(1..4)                 equality
#   41..44: skbe(1..4)                 equality
#   45..48: doppf(1..4)                equality
function mesh_model(; T = Float64, backend = nothing, kwargs...)
    pi_val  = T(acos(-1.0))
    omega1  = T(1.0e3)
    omega2  = T(-1.0e3)

    NV = 41

    # ── Variable index functions (CUTEst ordering) ──────────────────────────
    # Angle groups (in CUTEst order):
    # i=4: delta4=1, gamma4=2, beta4=3, alpha4=4
    # deltamin=5
    # i=3: delta3=6, gamma3=7, beta3=8, alpha3=9
    # f1=10
    # i=2: delta2=11, gamma2=12, beta2=13, alpha2=14
    # f3=15
    # i=1: delta1=16, gamma1=17, beta1=18, alpha1=19
    # l8=20, l7=21, l6=22, l5=23, l4=24, l3=25, l2=26, l1=27
    # y5=28, x5=29
    # Fixed: y1=30, y2=31, y3=32, x3=33, x4=34, x1=35, x2=36, y4=37
    # f4=38, fmax=39, f2=40, fmin=41

    # Group ordering by i: groups ordered i=4,3,2,1 with interleaved f1/f3/deltamin
    # alpha(i) positions: α[1]=19, α[2]=14, α[3]=9, α[4]=4
    α = (19, 14, 9, 4)
    β = (18, 13, 8, 3)
    γ = (17, 12, 7, 2)
    δ = (16, 11, 6, 1)

    # f(i) positions: f1=10, f2=40, f3=15, f4=38
    fi = (10, 40, 15, 38)

    # l(i) positions (reversed): l1=27, l2=26, l3=25, l4=24, l5=23, l6=22, l7=21, l8=20
    li(i) = 28 - i   # li(1)=27, li(2)=26, ..., li(8)=20

    # Scalar positions
    IDX_DMIN = 5     # deltamin
    IDX_FMIN = 41    # fmin
    IDX_FMAX = 39    # fmax
    IDX_Y5   = 28
    IDX_X5   = 29

    # Fixed variable positions
    IDX_Y1 = 30; IDX_Y2 = 31; IDX_Y3 = 32
    IDX_X3 = 33; IDX_X4 = 34
    IDX_X1 = 35; IDX_X2 = 36; IDX_Y4 = 37

    lvar = zeros(T, NV)
    uvar = fill(T(Inf), NV)

    # Angles [0, pi]
    for i in 1:4
        uvar[α[i]] = pi_val
        uvar[β[i]] = pi_val
        uvar[γ[i]] = pi_val
        uvar[δ[i]] = pi_val
    end
    uvar[IDX_DMIN] = pi_val

    # x5, y5: free
    lvar[IDX_X5] = T(-Inf); uvar[IDX_X5] = T(Inf)
    lvar[IDX_Y5] = T(-Inf); uvar[IDX_Y5] = T(Inf)

    # Fixed boundary nodes
    lvar[IDX_X1] = T(0); uvar[IDX_X1] = T(0)
    lvar[IDX_Y1] = T(0); uvar[IDX_Y1] = T(0)
    lvar[IDX_X2] = T(0); uvar[IDX_X2] = T(0)
    lvar[IDX_Y2] = T(1); uvar[IDX_Y2] = T(1)
    lvar[IDX_X3] = T(1); uvar[IDX_X3] = T(1)
    lvar[IDX_Y3] = T(1); uvar[IDX_Y3] = T(1)
    lvar[IDX_X4] = T(1); uvar[IDX_X4] = T(1)
    lvar[IDX_Y4] = T(0); uvar[IDX_Y4] = T(0)

    # Start point
    x0 = zeros(T, NV)
    # Fixed nodes
    x0[IDX_X1] = T(0);       x0[IDX_Y1] = T(0)
    x0[IDX_X2] = T(0);       x0[IDX_Y2] = T(1)
    x0[IDX_X3] = T(1);       x0[IDX_Y3] = T(1)
    x0[IDX_X4] = T(1);       x0[IDX_Y4] = T(0)
    x0[IDX_X5] = T(0.350000); x0[IDX_Y5] = T(0.606218)
    # l values
    x0[li(1)] = T(0.700000);  x0[li(2)] = T(0.526844)
    x0[li(3)] = T(1.000000);  x0[li(4)] = T(0.759977)
    x0[li(5)] = T(1.000000);  x0[li(6)] = T(0.888819)
    x0[li(7)] = T(1.000000);  x0[li(8)] = T(1.000000)
    # Angles
    x0[α[1]] = T(0.523599);  x0[β[1]] = T(1.891392)
    x0[γ[1]] = T(0.726602);  x0[δ[1]] = T(0.523599); x0[fi[1]] = T(0.350000)
    x0[α[2]] = T(0.844195);  x0[β[2]] = T(1.752711)
    x0[γ[2]] = T(0.544687);  x0[δ[2]] = T(0.544687); x0[fi[2]] = T(0.393782)
    x0[α[3]] = T(1.026109);  x0[β[3]] = T(1.295247)
    x0[γ[3]] = T(0.820236);  x0[δ[3]] = T(0.820236); x0[fi[3]] = T(0.650000)
    x0[α[4]] = T(0.750560);  x0[β[4]] = T(1.343835)
    x0[γ[4]] = T(1.047198);  x0[δ[4]] = T(0.750560); x0[fi[4]] = T(0.606218)
    x0[IDX_DMIN] = T(0.523599)
    x0[IDX_FMIN] = T(0.350000)
    x0[IDX_FMAX] = T(0.650000)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # ── Objective ──────────────────────────────────────────────────────────
    # obj1: deltamin^2 / omega1 (L2 group, SCALE=omega1=1000)
    ExaModels.@add_obj(c, x[IDX_DMIN]^2 / omega1)

    # obj2: (fmax-fmin)^2 / omega2 (L2 group, SCALE=omega2=-1000)
    ExaModels.@add_obj(c, (x[IDX_FMAX] - x[IDX_FMIN])^2 / omega2)

    # obj3: identity group, SCALE cancels → contribution = v_g
    # diffsq element: F = (winkel - minwi)^2
    for i in 1:4
        ExaModels.@add_obj(c, (x[α[i]] - x[δ[i]])^2)  # aldsq(i)
        ExaModels.@add_obj(c, (x[β[i]] - x[δ[i]])^2)  # bedsq(i)
        ExaModels.@add_obj(c, (x[γ[i]] - x[δ[i]])^2)  # gadsq(i)
    end

    # ── Constraints (CUTEst ordering: linear first, then nonlinear) ─────────

    # 1..4: wisum(i): alpha(i)+beta(i)+gamma(i) - pi = 0  (linear equality)
    ExaModels.@add_con(c, x[α[1]] + x[β[1]] + x[γ[1]] - pi_val)
    ExaModels.@add_con(c, x[α[2]] + x[β[2]] + x[γ[2]] - pi_val)
    ExaModels.@add_con(c, x[α[3]] + x[β[3]] + x[γ[3]] - pi_val)
    ExaModels.@add_con(c, x[α[4]] + x[β[4]] + x[γ[4]] - pi_val)

    # 5..8: alphd(i): alpha(i) - delta(i) >= 0
    ExaModels.@add_con(c, x[α[1]] - x[δ[1]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[α[2]] - x[δ[2]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[α[3]] - x[δ[3]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[α[4]] - x[δ[4]]; lcon = T(0), ucon = T(Inf))

    # 9..12: betad(i): beta(i) - delta(i) >= 0
    ExaModels.@add_con(c, x[β[1]] - x[δ[1]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[β[2]] - x[δ[2]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[β[3]] - x[δ[3]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[β[4]] - x[δ[4]]; lcon = T(0), ucon = T(Inf))

    # 13..16: gammd(i): gamma(i) - delta(i) >= 0
    ExaModels.@add_con(c, x[γ[1]] - x[δ[1]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[γ[2]] - x[δ[2]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[γ[3]] - x[δ[3]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[γ[4]] - x[δ[4]]; lcon = T(0), ucon = T(Inf))

    # 17..20: deltd(i): delta(i) - deltamin >= 0
    ExaModels.@add_con(c, x[δ[1]] - x[IDX_DMIN]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[δ[2]] - x[IDX_DMIN]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[δ[3]] - x[IDX_DMIN]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[δ[4]] - x[IDX_DMIN]; lcon = T(0), ucon = T(Inf))

    # 21..24: fmind(i): f(i) - fmin >= 0
    ExaModels.@add_con(c, x[fi[1]] - x[IDX_FMIN]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[fi[2]] - x[IDX_FMIN]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[fi[3]] - x[IDX_FMIN]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[fi[4]] - x[IDX_FMIN]; lcon = T(0), ucon = T(Inf))

    # 25..28: fmaxd(i): fmax - f(i) >= 0
    ExaModels.@add_con(c, x[IDX_FMAX] - x[fi[1]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[IDX_FMAX] - x[fi[2]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[IDX_FMAX] - x[fi[3]]; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[IDX_FMAX] - x[fi[4]]; lcon = T(0), ucon = T(Inf))

    # 29..36: seit(1..8): l_k^2 - (xA-xB)^2 - (yA-yB)^2 = 0  (laenge element)
    # laeng1: l1, (x1,y1)-(x5,y5)
    ExaModels.@add_con(c, x[li(1)]^2 - (x[IDX_X1]-x[IDX_X5])^2 - (x[IDX_Y1]-x[IDX_Y5])^2)
    # laeng2: l2, (x5,y5)-(x2,y2)
    ExaModels.@add_con(c, x[li(2)]^2 - (x[IDX_X5]-x[IDX_X2])^2 - (x[IDX_Y5]-x[IDX_Y2])^2)
    # laeng3: l3, (x2,y2)-(x1,y1)
    ExaModels.@add_con(c, x[li(3)]^2 - (x[IDX_X2]-x[IDX_X1])^2 - (x[IDX_Y2]-x[IDX_Y1])^2)
    # laeng4: l4, (x5,y5)-(x3,y3)
    ExaModels.@add_con(c, x[li(4)]^2 - (x[IDX_X5]-x[IDX_X3])^2 - (x[IDX_Y5]-x[IDX_Y3])^2)
    # laeng5: l5, (x3,y3)-(x2,y2)
    ExaModels.@add_con(c, x[li(5)]^2 - (x[IDX_X3]-x[IDX_X2])^2 - (x[IDX_Y3]-x[IDX_Y2])^2)
    # laeng6: l6, (x5,y5)-(x4,y4)
    ExaModels.@add_con(c, x[li(6)]^2 - (x[IDX_X5]-x[IDX_X4])^2 - (x[IDX_Y5]-x[IDX_Y4])^2)
    # laeng7: l7, (x4,y4)-(x3,y3)
    ExaModels.@add_con(c, x[li(7)]^2 - (x[IDX_X4]-x[IDX_X3])^2 - (x[IDX_Y4]-x[IDX_Y3])^2)
    # laeng8: l8, (x1,y1)-(x4,y4)
    ExaModels.@add_con(c, x[li(8)]^2 - (x[IDX_X1]-x[IDX_X4])^2 - (x[IDX_Y1]-x[IDX_Y4])^2)

    # 37..40: skal(i): sklprod(sal_i) - cosprod(cal_i) = 0
    # sal1=(x2,y2,x1,y1,x5,y5), cal1=(l3,l1,alpha1): dot(p2-p1, p5-p1) - l3*l1*cos(alpha1)
    ExaModels.@add_con(c,
        (x[IDX_X2]-x[IDX_X1])*(x[IDX_X5]-x[IDX_X1]) + (x[IDX_Y2]-x[IDX_Y1])*(x[IDX_Y5]-x[IDX_Y1])
        - x[li(3)]*x[li(1)]*cos(x[α[1]]))
    # sal2=(x3,y3,x2,y2,x5,y5), cal2=(l5,l2,alpha2)
    ExaModels.@add_con(c,
        (x[IDX_X3]-x[IDX_X2])*(x[IDX_X5]-x[IDX_X2]) + (x[IDX_Y3]-x[IDX_Y2])*(x[IDX_Y5]-x[IDX_Y2])
        - x[li(5)]*x[li(2)]*cos(x[α[2]]))
    # sal3=(x4,y4,x3,y3,x5,y5), cal3=(l7,l4,alpha3)
    ExaModels.@add_con(c,
        (x[IDX_X4]-x[IDX_X3])*(x[IDX_X5]-x[IDX_X3]) + (x[IDX_Y4]-x[IDX_Y3])*(x[IDX_Y5]-x[IDX_Y3])
        - x[li(7)]*x[li(4)]*cos(x[α[3]]))
    # sal4=(x1,y1,x4,y4,x5,y5), cal4=(l8,l6,alpha4)
    ExaModels.@add_con(c,
        (x[IDX_X1]-x[IDX_X4])*(x[IDX_X5]-x[IDX_X4]) + (x[IDX_Y1]-x[IDX_Y4])*(x[IDX_Y5]-x[IDX_Y4])
        - x[li(8)]*x[li(6)]*cos(x[α[4]]))

    # 41..44: skbe(i): sklprod(sbe_i) - cosprod(cbe_i) = 0
    # sbe1=(x1,y1,x5,y5,x2,y2), cbe1=(l1,l2,beta1): dot(p1-p5,p2-p5) - l1*l2*cos(beta1)
    ExaModels.@add_con(c,
        (x[IDX_X1]-x[IDX_X5])*(x[IDX_X2]-x[IDX_X5]) + (x[IDX_Y1]-x[IDX_Y5])*(x[IDX_Y2]-x[IDX_Y5])
        - x[li(1)]*x[li(2)]*cos(x[β[1]]))
    # sbe2=(x2,y2,x5,y5,x3,y3), cbe2=(l2,l4,beta2)
    ExaModels.@add_con(c,
        (x[IDX_X2]-x[IDX_X5])*(x[IDX_X3]-x[IDX_X5]) + (x[IDX_Y2]-x[IDX_Y5])*(x[IDX_Y3]-x[IDX_Y5])
        - x[li(2)]*x[li(4)]*cos(x[β[2]]))
    # sbe3=(x3,y3,x5,y5,x4,y4), cbe3=(l4,l6,beta3)
    ExaModels.@add_con(c,
        (x[IDX_X3]-x[IDX_X5])*(x[IDX_X4]-x[IDX_X5]) + (x[IDX_Y3]-x[IDX_Y5])*(x[IDX_Y4]-x[IDX_Y5])
        - x[li(4)]*x[li(6)]*cos(x[β[3]]))
    # sbe4=(x4,y4,x5,y5,x1,y1), cbe4=(l6,l1,beta4)
    ExaModels.@add_con(c,
        (x[IDX_X4]-x[IDX_X5])*(x[IDX_X1]-x[IDX_X5]) + (x[IDX_Y4]-x[IDX_Y5])*(x[IDX_Y1]-x[IDX_Y5])
        - x[li(6)]*x[li(1)]*cos(x[β[4]]))

    # 45..48: doppf(i): -f(i) + vekprod(flae_i) = 0
    # vekprod F = (vp3-vp2)×(vp1-vp2) = (vp3x-vp2x)*(vp1y-vp2y) - (vp1x-vp2x)*(vp3y-vp2y)
    # flae1: vp1=(x1,y1), vp2=(x5,y5), vp3=(x2,y2)
    ExaModels.@add_con(c,
        -x[fi[1]]
        + (x[IDX_X2]-x[IDX_X5])*(x[IDX_Y1]-x[IDX_Y5]) - (x[IDX_X1]-x[IDX_X5])*(x[IDX_Y2]-x[IDX_Y5]))
    # flae2: vp1=(x2,y2), vp2=(x5,y5), vp3=(x3,y3)
    ExaModels.@add_con(c,
        -x[fi[2]]
        + (x[IDX_X3]-x[IDX_X5])*(x[IDX_Y2]-x[IDX_Y5]) - (x[IDX_X2]-x[IDX_X5])*(x[IDX_Y3]-x[IDX_Y5]))
    # flae3: vp1=(x3,y3), vp2=(x5,y5), vp3=(x4,y4)
    ExaModels.@add_con(c,
        -x[fi[3]]
        + (x[IDX_X4]-x[IDX_X5])*(x[IDX_Y3]-x[IDX_Y5]) - (x[IDX_X3]-x[IDX_X5])*(x[IDX_Y4]-x[IDX_Y5]))
    # flae4: vp1=(x4,y4), vp2=(x5,y5), vp3=(x1,y1)
    ExaModels.@add_con(c,
        -x[fi[4]]
        + (x[IDX_X1]-x[IDX_X5])*(x[IDX_Y4]-x[IDX_Y5]) - (x[IDX_X4]-x[IDX_X5])*(x[IDX_Y1]-x[IDX_Y5]))

    return ExaModels.ExaModel(c; kwargs...)
end
