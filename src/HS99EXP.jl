# HS99EXP: Expanded form of Hock-Schittkowski problem 99
# 31 variables, 21 equality constraints
# Variables: X(1..7)→x[1..7], R(1..8)→x[8..15], Q(1..8)→x[16..23], S(1..8)→x[24..31]
# R(1)=Q(1)=S(1)=0 (fixed), X(I)∈[0,1.58], rest free
# Start: X(I)=0.5 for I=1..7, all others=0
#
# Objective: min -R(8)^2 = min -(x[15])^2
# (L2 group OBJ with SCALE=-1: contribution = v^2/(-1) = -R(8)^2)
#
# Equality constraints for I=2..8:
#   R(I)DEF: R(I) - R(I-1) - A(I)*DT(I)*cos(X(I-1)) = 0
#   Q(I)DEF: Q(I) - Q(I-1) - S(I-1)*DT(I) - A(I)*(DT(I)^2/2)*sin(X(I-1)) = RHS_Q(I)
#   S(I)DEF: S(I) - S(I-1) - A(I)*DT(I)*sin(X(I-1)) = RHS_S(I)
# where for I=2..7: RHS_Q(I) = (DT(I)^2/2)*B, RHS_S(I) = DT(I)*B
#       for I=8:    RHS_Q(8) = 100000.0, RHS_S(8) = 1000.0
function hs99exp_model(; T = Float64, backend = nothing, kwargs...)
    T_vals = [T(0), T(25), T(50), T(100), T(150), T(200), T(290), T(380)]
    A_vals = [T(0), T(50), T(50), T(75), T(75), T(75), T(100), T(100)]
    B = T(32)

    DT = T[T_vals[i] - T_vals[i-1] for i in 2:8]   # DT(I) for I=2..8, length=7
    DTSQ2 = T[DT[i]^2 / T(2) for i in 1:7]          # DT(I)^2/2

    # RHS for Q and S constraints (I=2..8, so index i=1..7 corresponds to I=2..8)
    RHS_Q = T[i <= 6 ? DTSQ2[i] * B : T(100000) for i in 1:7]
    RHS_S = T[i <= 6 ? DT[i] * B : T(1000) for i in 1:7]

    # Weights W = A(I)*DT(I) for I=2..8 (element coeff for R and S constraints)
    # W_R[i] = A_vals[i+1] * DT[i] for i=1..7 (I=2..8)
    W_R = T[A_vals[i+1] * DT[i] for i in 1:7]
    W_SQ2 = T[A_vals[i+1] * DTSQ2[i] for i in 1:7]  # A(I)*DT(I)^2/2

    nvar = 31
    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    start = zeros(T, nvar)

    # X(I) in [0, 1.58] with start 0.5
    for i in 1:7
        lvar[i] = T(0)
        uvar[i] = T(1.58)
        start[i] = T(0.5)
    end
    # Fix R(1)=Q(1)=S(1)=0: positions 8, 16, 24
    lvar[8] = T(0);  uvar[8] = T(0)
    lvar[16] = T(0); uvar[16] = T(0)
    lvar[24] = T(0); uvar[24] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Objective: min -(x[15])^2 = min -R(8)^2
    ExaModels.@add_obj(c, -x[15]^2)

    # Parameters for element weights
    ExaModels.@add_par(c, WR, W_R)
    ExaModels.@add_par(c, WSQ2, W_SQ2)
    ExaModels.@add_par(c, DTp, DT)
    ExaModels.@add_par(c, RHSQ, RHS_Q)
    ExaModels.@add_par(c, RHSS, RHS_S)

    # R(I)DEF: R(I) - R(I-1) - A(I)*DT(I)*cos(X(I-1)) = 0
    # For I=2..8: R(I)=x[7+I], R(I-1)=x[6+I], X(I-1)=x[I-1]
    # i (parameter index 1..7) corresponds to I=i+1: R(I)=x[8+i], R(I-1)=x[7+i], X(I-1)=x[i]
    ExaModels.@add_con(c,
        x[8 + i] - x[7 + i] - WR[i] * cos(x[i])
        for i in 1:7)

    # Q(I)DEF: Q(I) - Q(I-1) - S(I-1)*DT(I) - A(I)*(DT(I)^2/2)*sin(X(I-1)) = RHS_Q(I)
    # Q(I)=x[15+i], Q(I-1)=x[14+i], S(I-1)=x[22+i], X(I-1)=x[i]
    ExaModels.@add_con(c,
        x[15 + i] - x[14 + i] - x[22 + i] * DTp[i] - WSQ2[i] * sin(x[i]) - RHSQ[i]
        for i in 1:7)

    # S(I)DEF: S(I) - S(I-1) - A(I)*DT(I)*sin(X(I-1)) = RHS_S(I)
    # S(I)=x[23+i], S(I-1)=x[22+i], X(I-1)=x[i]
    ExaModels.@add_con(c,
        x[23 + i] - x[22 + i] - WR[i] * sin(x[i]) - RHSS[i]
        for i in 1:7)

    return ExaModels.ExaModel(c; kwargs...)
end
