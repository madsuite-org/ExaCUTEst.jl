# CATENARY: erroneous but interesting 3D catenary chain problem
# Differs from CATENA by a typo: EZ uses V=X(I) instead of V=Z(I)
# N+1=1000 beams of length BL=1, N+2=1001 joints (vertices)
# Variables: X(I), Y(I), Z(I) for I=0..N+1, interleaved: x[3I+1]=X(I), x[3I+2]=Y(I), x[3I+3]=Z(I)
# Total vars: 3*(N+2)=3003
# Fixed: X(0)=Y(0)=Z(0)=0, X(N+1)=LENGTH=600; all others free
# Objective (linear): MG/2*Y(0) + MG*Y(1..N) + MG/2*Y(N+1)
#   MG=MASS*GAMMA=0.5*9.81=4.905, MG/2=2.4525
# Constraints C(I) for I=1..N+1 (equality):
#   (X(I)-X(I-1))^2 + (Y(I)-Y(I-1))^2 + (X(I)-Z(I-1))^2 - BLSQ = 0
#   Note: EZ uses V=X(I) (typo for Z(I)), W=Z(I-1)
# Start: X(I)=TMP=0.6 (constant, not I*TMP), Y(I)=0, Z(I)=0 for I=1..N+1
function catenary_model(; T = Float64, backend = nothing, kwargs...)
    Np1 = 1000          # N+1 = number of beams
    N = Np1 - 1         # N = 999
    GAMMA = T(9.81)
    TMASS = T(500.0)
    BL = T(1.0)
    FRACT = T(0.6)
    LENGTH = BL * Np1 * FRACT   # = 600.0
    MASS = TMASS / Np1           # = 0.5
    MG = MASS * GAMMA            # = 4.905
    MG2 = MG / 2                 # = 2.4525
    BLSQ = BL * BL               # = 1.0
    TMP = LENGTH / Np1           # = 0.6

    # 3*(N+2) = 3*1001 = 3003 variables
    # Index mapping: X(I) -> x[3I+1], Y(I) -> x[3I+2], Z(I) -> x[3I+3] (0-indexed I)

    nv = 3 * (Np1 + 1)   # 3003
    lvar = fill(T(-Inf), nv)
    uvar = fill(T(Inf), nv)
    start = zeros(T, nv)

    # Fix X(0)=Y(0)=Z(0)=0  (indices 1,2,3)
    lvar[1] = T(0); uvar[1] = T(0)
    lvar[2] = T(0); uvar[2] = T(0)
    lvar[3] = T(0); uvar[3] = T(0)

    # Fix X(N+1)=LENGTH  (index 3*(N+1)+1 = 3001)
    lvar[3*Np1+1] = LENGTH; uvar[3*Np1+1] = LENGTH
    start[3*Np1+1] = LENGTH

    # Start point: X(I)=TMP (constant 0.6), Y(I)=0, Z(I)=0 for I=1..N+1
    for i in 1:Np1
        start[3*i+1] = TMP   # X(I) = TMP (constant, same for all I)
        # Y(I) = 0, Z(I) = 0 (already zeroed)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Objective: MG/2*Y(0) + MG*Y(I for I=1..N) + MG/2*Y(N+1)
    # Y(0) = x[2], Y(N+1) = x[3*(N+1)+2] = x[3002]
    ExaModels.@add_obj(c, MG2 * x[2])                 # Y(0)
    ExaModels.@add_obj(c, MG * x[3*i+2] for i in 1:N)  # Y(1)..Y(N)
    ExaModels.@add_obj(c, MG2 * x[3*Np1+2])            # Y(N+1) = x[3002]

    # Constraints C(I) for I=1..N+1:
    # (X(I)-X(I-1))^2 + (Y(I)-Y(I-1))^2 + (X(I)-Z(I-1))^2 - BLSQ = 0
    # X(I)=x[3I+1], X(I-1)=x[3I-2], Y(I)=x[3I+2], Y(I-1)=x[3I-1], Z(I-1)=x[3I]
    # EZ uses V=X(I)=x[3I+1] and W=Z(I-1)=x[3I]
    ExaModels.@add_con(
        c,
        (x[3*i+1] - x[3*i-2])^2 + (x[3*i+2] - x[3*i-1])^2 + (x[3*i+1] - x[3*i])^2 - BLSQ
        for i in 1:Np1
    )

    return ExaModels.ExaModel(c; kwargs...)
end
