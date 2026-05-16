# SSC: Steady-State Combustion 2D PDE, NX=NY=70 default
# Source: Averick & More (1993); SIF input by Ph. Toint, 1993
# classification OXR2-MY-V-0 (unconstrained, boundary vars fixed)
#
# Variables: X(I,J) for I=0..NX+1, J=0..NY+1 (boundary fixed at 0)
# Julia index: x[I*(NY+2) + J + 1]
#
# Objective (sum of N-type groups):
#   FL(I,J) for I=0..NX, J=0..NY:
#     HY/4HX*(X(I+1,J)-X(I,J))^2 + HX/4HY*(X(I,J+1)-X(I,J))^2
#     - LHXHY/6*(exp(X(I,J))+exp(X(I+1,J))+exp(X(I,J+1)))
#   FU(I,J) for I=1..NX+1, J=1..NY+1:
#     HY/4HX*(X(I-1,J)-X(I,J))^2 + HX/4HY*(X(I,J-1)-X(I,J))^2
#     - LHXHY/6*(exp(X(I,J))+exp(X(I-1,J))+exp(X(I,J-1)))
function ssc_model(NX = 70, NY = 70; T = Float64, backend = nothing, kwargs...)
    LAMBDA = T(2.0)
    HX = T(1) / T(NX + 1)
    HY = T(1) / T(NY + 1)
    HY4HX = HY / (T(4) * HX)   # HY/(4*HX)
    HX4HY = HX / (T(4) * HY)   # HX/(4*HY)
    HXHY  = HX * HY
    LHXHY6 = LAMBDA * HXHY / T(6)  # LHXHY/6

    # Variable index: X(I,J) = x[I*(NY+2) + J + 1]
    xidx = (i, j) -> i * (NY + 2) + j + 1

    nv = (NX + 2) * (NY + 2)
    lvar = fill(T(-Inf), nv)
    uvar = fill(T(Inf), nv)
    start = zeros(T, nv)

    # Fix boundary variables at 0
    for I in 0:NX+1
        lvar[xidx(I, 0)] = T(0); uvar[xidx(I, 0)] = T(0)
        lvar[xidx(I, NY + 1)] = T(0); uvar[xidx(I, NY + 1)] = T(0)
    end
    for J in 1:NY
        lvar[xidx(0, J)] = T(0); uvar[xidx(0, J)] = T(0)
        lvar[xidx(NX + 1, J)] = T(0); uvar[xidx(NX + 1, J)] = T(0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Lower triangle FL(I,J) for I=0..NX, J=0..NY:
    fl_data = [(Iij = xidx(I, J), Ip1j = xidx(I + 1, J), Ijp1 = xidx(I, J + 1))
               for I in 0:NX for J in 0:NY]
    ExaModels.@add_obj(c,
        HY4HX * (x[d.Ip1j] - x[d.Iij])^2 +
        HX4HY * (x[d.Ijp1] - x[d.Iij])^2 -
        LHXHY6 * (exp(x[d.Iij]) + exp(x[d.Ip1j]) + exp(x[d.Ijp1]))
        for d in fl_data)

    # Upper triangle FU(I,J) for I=1..NX+1, J=1..NY+1:
    fu_data = [(Iij = xidx(I, J), Im1j = xidx(I - 1, J), Ijm1 = xidx(I, J - 1))
               for I in 1:NX+1 for J in 1:NY+1]
    ExaModels.@add_obj(c,
        HY4HX * (x[d.Im1j] - x[d.Iij])^2 +
        HX4HY * (x[d.Ijm1] - x[d.Iij])^2 -
        LHXHY6 * (exp(x[d.Iij]) + exp(x[d.Im1j]) + exp(x[d.Ijm1]))
        for d in fu_data)

    return ExaModels.ExaModel(c; kwargs...)
end
