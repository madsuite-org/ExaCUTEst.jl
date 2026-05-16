# STEERING: Particle steering optimal control (COPS problem 9), NH=400 default
# Source: Dolan & More (2000); SIF input by N. Gould, 2000
# Minimize final time for a particle under constant thrust to reach target position/velocity
#
# Variables: H (step), U(0..NH), Y1(0..NH), Y2(0..NH), Y3(0..NH), Y4(0..NH)
# Julia ordering: x[1]=H; for I=0..NH: U(I)=x[5I+2], Y1(I)=x[5I+3], Y2(I)=x[5I+4],
#                                       Y3(I)=x[5I+5], Y4(I)=x[5I+6]
#
# Objective: NH * H (linear, N type)
#
# Constraints for I=0..NH-1 (E type):
#   C1(I): Y1(I) - Y1(I+1) + 0.5*H*Y3(I) + 0.5*H*Y3(I+1) = 0
#   C2(I): Y2(I) - Y2(I+1) + 0.5*H*Y4(I) + 0.5*H*Y4(I+1) = 0
#   C3(I): Y3(I) - Y3(I+1) + (A/2)*H*cos(U(I)) + (A/2)*H*cos(U(I+1)) = 0
#   C4(I): Y4(I) - Y4(I+1) + (A/2)*H*sin(U(I)) + (A/2)*H*sin(U(I+1)) = 0
#
# Bounds: H >= 0, U(I) in [-PI/2, PI/2], Y1/Y2/Y3/Y4 free
#         Y1(0)=Y2(0)=Y3(0)=Y4(0)=0 fixed, Y2(NH)=5, Y3(NH)=45, Y4(NH)=0 fixed
function steering_model(NH = 400; T = Float64, backend = nothing, kwargs...)
    A     = T(100.0)
    AH2   = A / T(2)     # A/2
    PI    = T(4) * atan(T(1))
    PI2   = PI / T(2)    # PI/2
    NH_T  = T(NH)

    nv = 1 + 5 * (NH + 1)
    lvar = fill(T(-Inf), nv)
    uvar = fill(T(Inf), nv)
    start = zeros(T, nv)

    # H >= 0
    lvar[1] = T(0); start[1] = T(1) / T(NH)

    for I in 0:NH
        base = 5 * I + 2
        # U(I) in [-PI/2, PI/2]
        lvar[base]     = -PI2;  uvar[base]     = PI2
        # Y1(I): free
        # Y2(I): free
        # Y3(I): free
        # Y4(I): free
        # Start values
        start[base]     = T(0)                    # U(I)
        start[base + 1] = T(0)                    # Y1(I)
        start[base + 2] = T(5) * T(I) / T(NH)    # Y2(I) = 5*I/NH
        start[base + 3] = T(45) * T(I) / T(NH)   # Y3(I) = 45*I/NH
        start[base + 4] = T(0)                    # Y4(I)
    end

    # Fixed boundary conditions
    # Y1(0)=0: x[3] fixed at 0
    lvar[3] = T(0); uvar[3] = T(0)
    # Y2(0)=0: x[4] fixed at 0
    lvar[4] = T(0); uvar[4] = T(0)
    # Y3(0)=0: x[5] fixed at 0
    lvar[5] = T(0); uvar[5] = T(0)
    # Y4(0)=0: x[6] fixed at 0
    lvar[6] = T(0); uvar[6] = T(0)
    # Y2(NH)=5: x[5*NH+4] fixed at 5
    lvar[5 * NH + 4] = T(5); uvar[5 * NH + 4] = T(5); start[5 * NH + 4] = T(5)
    # Y3(NH)=45: x[5*NH+5] fixed at 45
    lvar[5 * NH + 5] = T(45); uvar[5 * NH + 5] = T(45); start[5 * NH + 5] = T(45)
    # Y4(NH)=0: x[5*NH+6] fixed at 0
    lvar[5 * NH + 6] = T(0); uvar[5 * NH + 6] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Objective: NH * H
    ExaModels.@add_obj(c, NH_T * x[1])

    # Constraints: for I = 0..NH-1
    # U(I)   = x[5I+2], Y1(I)=x[5I+3], Y2(I)=x[5I+4], Y3(I)=x[5I+5], Y4(I)=x[5I+6]
    # U(I+1) = x[5(I+1)+2] = x[5I+7], Y1(I+1)=x[5I+8], Y2(I+1)=x[5I+9],
    #                                   Y3(I+1)=x[5I+10], Y4(I+1)=x[5I+11]
    # CUTEst order: I-outer, type-inner (C1(I),C2(I),C3(I),C4(I) for each I)
    for I in 0:NH-1
        # C1(I): Y1(I) - Y1(I+1) + 0.5*H*Y3(I) + 0.5*H*Y3(I+1) = 0
        ExaModels.@add_con(c,
            x[5 * I + 3] - x[5 * I + 8] +
            T(0.5) * x[1] * x[5 * I + 5] + T(0.5) * x[1] * x[5 * I + 10])
        # C2(I): Y2(I) - Y2(I+1) + 0.5*H*Y4(I) + 0.5*H*Y4(I+1) = 0
        ExaModels.@add_con(c,
            x[5 * I + 4] - x[5 * I + 9] +
            T(0.5) * x[1] * x[5 * I + 6] + T(0.5) * x[1] * x[5 * I + 11])
        # C3(I): Y3(I) - Y3(I+1) + (A/2)*H*cos(U(I)) + (A/2)*H*cos(U(I+1)) = 0
        ExaModels.@add_con(c,
            x[5 * I + 5] - x[5 * I + 10] +
            AH2 * x[1] * cos(x[5 * I + 2]) + AH2 * x[1] * cos(x[5 * I + 7]))
        # C4(I): Y4(I) - Y4(I+1) + (A/2)*H*sin(U(I)) + (A/2)*H*sin(U(I+1)) = 0
        ExaModels.@add_con(c,
            x[5 * I + 6] - x[5 * I + 11] +
            AH2 * x[1] * sin(x[5 * I + 2]) + AH2 * x[1] * sin(x[5 * I + 7]))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
