# ROBOTARM: Minimize time for robot arm motion (COPS problem 8), NH=400 default
# Source: Dolan & More (2000); SIF input by N. Gould, 2000
#
# Variables: STEP, then per I=0..NH: RHO,THE,PHI, RHO_dot,THE_dot,PHI_dot,
#                                     URHO,UTHE,UPHI, ITHE,IPHI
# Julia: x[1]=STEP; base(I)=11*I+2; RHO(I)=x[b], THE(I)=x[b+1], PHI(I)=x[b+2],
#        RHO.(I)=x[b+3], THE.(I)=x[b+4], PHI.(I)=x[b+5],
#        URHO(I)=x[b+6], UTHE(I)=x[b+7], UPHI(I)=x[b+8],
#        ITHE(I)=x[b+9], IPHI(I)=x[b+10]  where b=11*I+2
#
# Objective (N type): NH * STEP
#
# Constraints (E type):
#   RHO(J): -RHO(J)+RHO(J-1)+0.5*STEP*(RHO.(J)+RHO.(J-1)) = 0, J=1..NH
#   THE(J): -THE(J)+THE(J-1)+0.5*STEP*(THE.(J)+THE.(J-1)) = 0
#   PHI(J): -PHI(J)+PHI(J-1)+0.5*STEP*(PHI.(J)+PHI.(J-1)) = 0
#   RHO.(J): -RHO.(J)+RHO.(J-1)+(0.5/L)*STEP*(URHO(J)+URHO(J-1)) = 0
#   THE.(J): -THE.(J)+THE.(J-1)+0.5*STEP*(UTHE(J)/ITHE(J)+UTHE(J-1)/ITHE(J-1)) = 0
#   PHI.(J): -PHI.(J)+PHI.(J-1)+0.5*STEP*(UPHI(J)/IPHI(J)+UPHI(J-1)/IPHI(J-1)) = 0
#   ITHE(I): -ITHE(I)+sin(PHI(I))^2*((L-RHO(I))^3+RHO(I)^3)/3 = 0, I=0..NH
#   IPHI(I): -IPHI(I)+((L-RHO(I))^3+RHO(I)^3)/3 = 0, I=0..NH
function robotarm_model(NH = 400; T = Float64, backend = nothing, kwargs...)
    L        = T(5.0)
    MAXURHO  = T(1.0)
    MAXUTHE  = T(1.0)
    MAXUPHI  = T(1.0)
    PI       = T(4) * atan(T(1))
    INV2L    = T(1) / (T(2) * L)  # 0.5/L
    NH_T     = T(NH)
    ONE3     = T(1) / T(3)

    nv = 1 + 11 * (NH + 1)
    lvar = fill(T(-Inf), nv)
    uvar = fill(T(Inf), nv)
    start = zeros(T, nv)

    # STEP >= 0
    lvar[1] = T(0)
    start[1] = T(1) / T(NH)

    for I in 0:NH
        b = 11 * I + 2
        # RHO(I) in [0, L=5]
        lvar[b] = T(0); uvar[b] = L
        # THE(I) in [-PI, PI]
        lvar[b + 1] = -PI; uvar[b + 1] = PI
        # PHI(I) in [0, PI]
        lvar[b + 2] = T(0); uvar[b + 2] = PI
        # URHO(I) in [-MAXURHO, MAXURHO]
        lvar[b + 6] = -MAXURHO; uvar[b + 6] = MAXURHO
        # UTHE(I) in [-MAXUTHE, MAXUTHE]
        lvar[b + 7] = -MAXUTHE; uvar[b + 7] = MAXUTHE
        # UPHI(I) in [-MAXUPHI, MAXUPHI]
        lvar[b + 8] = -MAXUPHI; uvar[b + 8] = MAXUPHI

        # Start values
        RI = T(I) / T(NH)
        THEI  = T(2) * PI / T(3) * RI^2
        THE_I = T(4) * PI / T(3) * RI
        PHII  = PI / T(4)
        RHOI  = T(4.5)
        T1 = L - RHOI
        IPHII = (T1^3 + RHOI^3) / T(3)
        SINP  = sin(PHII)
        ITHEI = SINP^2 * IPHII

        start[b]     = RHOI
        start[b + 1] = THEI
        start[b + 2] = PHII
        start[b + 4] = THE_I
        start[b + 9] = ITHEI
        start[b + 10] = IPHII
    end

    # Fixed boundary conditions
    # RHO(0)=4.5, THE(0)=0, PHI(0)=PI/4
    b0 = 2
    lvar[b0] = T(4.5); uvar[b0] = T(4.5); start[b0] = T(4.5)
    lvar[b0 + 1] = T(0); uvar[b0 + 1] = T(0)
    lvar[b0 + 2] = PI / T(4); uvar[b0 + 2] = PI / T(4); start[b0 + 2] = PI / T(4)
    # RHO.(0)=0, THE.(0)=0, PHI.(0)=0
    lvar[b0 + 3] = T(0); uvar[b0 + 3] = T(0)
    lvar[b0 + 4] = T(0); uvar[b0 + 4] = T(0)
    lvar[b0 + 5] = T(0); uvar[b0 + 5] = T(0)

    bN = 11 * NH + 2
    # RHO(NH)=4.5, THE(NH)=2PI/3, PHI(NH)=PI/4
    lvar[bN] = T(4.5); uvar[bN] = T(4.5); start[bN] = T(4.5)
    lvar[bN + 1] = T(2) * PI / T(3); uvar[bN + 1] = T(2) * PI / T(3);
    start[bN + 1] = T(2) * PI / T(3)
    lvar[bN + 2] = PI / T(4); uvar[bN + 2] = PI / T(4); start[bN + 2] = PI / T(4)
    # RHO.(NH)=0, THE.(NH)=0, PHI.(NH)=0
    lvar[bN + 3] = T(0); uvar[bN + 3] = T(0)
    lvar[bN + 4] = T(0); uvar[bN + 4] = T(0)
    lvar[bN + 5] = T(0); uvar[bN + 5] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Objective: NH * STEP
    ExaModels.@add_obj(c, NH_T * x[1])

    data = [(J = J,) for J in 1:NH]
    data0N = [(I = I,) for I in 0:NH]

    # RHO(J): -RHO(J)+RHO(J-1)+0.5*STEP*(RHO.(J)+RHO.(J-1)) = 0
    ExaModels.@add_con(c,
        -x[11 * d.J + 2] + x[11 * (d.J - 1) + 2] +
        T(0.5) * x[1] * (x[11 * d.J + 5] + x[11 * (d.J - 1) + 5])
        for d in data)

    # THE(J): -THE(J)+THE(J-1)+0.5*STEP*(THE.(J)+THE.(J-1)) = 0
    ExaModels.@add_con(c,
        -x[11 * d.J + 3] + x[11 * (d.J - 1) + 3] +
        T(0.5) * x[1] * (x[11 * d.J + 6] + x[11 * (d.J - 1) + 6])
        for d in data)

    # PHI(J): -PHI(J)+PHI(J-1)+0.5*STEP*(PHI.(J)+PHI.(J-1)) = 0
    ExaModels.@add_con(c,
        -x[11 * d.J + 4] + x[11 * (d.J - 1) + 4] +
        T(0.5) * x[1] * (x[11 * d.J + 7] + x[11 * (d.J - 1) + 7])
        for d in data)

    # RHO.(J): -RHO.(J)+RHO.(J-1)+INV2L*STEP*(URHO(J)+URHO(J-1)) = 0
    ExaModels.@add_con(c,
        -x[11 * d.J + 5] + x[11 * (d.J - 1) + 5] +
        INV2L * x[1] * (x[11 * d.J + 8] + x[11 * (d.J - 1) + 8])
        for d in data)

    # THE.(J): -THE.(J)+THE.(J-1)+0.5*STEP*(UTHE(J)/ITHE(J)+UTHE(J-1)/ITHE(J-1)) = 0
    ExaModels.@add_con(c,
        -x[11 * d.J + 6] + x[11 * (d.J - 1) + 6] +
        T(0.5) * x[1] * (x[11 * d.J + 9] / x[11 * d.J + 11] +
                          x[11 * (d.J - 1) + 9] / x[11 * (d.J - 1) + 11])
        for d in data)

    # PHI.(J): -PHI.(J)+PHI.(J-1)+0.5*STEP*(UPHI(J)/IPHI(J)+UPHI(J-1)/IPHI(J-1)) = 0
    ExaModels.@add_con(c,
        -x[11 * d.J + 7] + x[11 * (d.J - 1) + 7] +
        T(0.5) * x[1] * (x[11 * d.J + 10] / x[11 * d.J + 12] +
                          x[11 * (d.J - 1) + 10] / x[11 * (d.J - 1) + 12])
        for d in data)

    # ITHE(I): -ITHE(I)+sin(PHI(I))^2*((L-RHO(I))^3+RHO(I)^3)/3 = 0
    ExaModels.@add_con(c,
        -x[11 * d.I + 11] +
        sin(x[11 * d.I + 4])^2 * ONE3 * ((L - x[11 * d.I + 2])^3 + x[11 * d.I + 2]^3)
        for d in data0N)

    # IPHI(I): -IPHI(I)+((L-RHO(I))^3+RHO(I)^3)/3 = 0
    ExaModels.@add_con(c,
        -x[11 * d.I + 12] +
        ONE3 * ((L - x[11 * d.I + 2])^3 + x[11 * d.I + 2]^3)
        for d in data0N)

    return ExaModels.ExaModel(c; kwargs...)
end
