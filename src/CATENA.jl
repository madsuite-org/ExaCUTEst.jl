# CATENA: classical 3D catenary hanging chain problem
# N+1=1000 beams of length BL=1, N+2=1001 joints (vertices)
# Variables: X(I), Y(I), Z(I) for I=0..N+1, interleaved: x[3I+1]=X(I), x[3I+2]=Y(I), x[3I+3]=Z(I)
# Total vars: 3*(N+2)=3003
# Fixed: X(0)=Y(0)=Z(0)=0, X(N+1)=LENGTH=600; all others free
# Objective (linear): MG/2*Y(0) + MG*Y(1..N) + MG/2*Y(N+1)
#   MG=MASS*GAMMA=0.5*9.81=4.905, MG/2=2.4525
# Constraints C(I) for I=1..N+1 (equality):
#   (X(I)-X(I-1))^2 + (Y(I)-Y(I-1))^2 + (Z(I)-Z(I-1))^2 - BLSQ = 0
#   BLSQ=BL^2=1
# Start: X(I)=I*LENGTH/(N+1)=I*0.6, Y(I)=-I*0.6, Z(I)=0 for I=1..N+1
function catena_model(; T = Float64, backend = nothing, kwargs...)
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
    # In 1-based Julia indices: for I=0..N+1:
    #   ix(I) = 3I+1,  iy(I) = 3I+2,  iz(I) = 3I+3
    # Total = 3*(N+2) where N+2 = N+1+1 = 1001, so 3003

    nv = 3 * (Np1 + 1)   # 3003
    lvar = fill(T(-Inf), nv)
    uvar = fill(T(Inf), nv)
    start = zeros(T, nv)

    # Fix X(0)=Y(0)=Z(0)=0  (indices 1,2,3)
    lvar[1] = T(0); uvar[1] = T(0)
    lvar[2] = T(0); uvar[2] = T(0)
    lvar[3] = T(0); uvar[3] = T(0)

    # Fix X(N+1)=LENGTH  (index 3*(N+1)+1 = 3*1000+1 = 3001)
    lvar[3*Np1+1] = LENGTH; uvar[3*Np1+1] = LENGTH
    start[3*Np1+1] = LENGTH

    # Start point: X(I)=TMP*I, Y(I)=-TMP*I, Z(I)=0 for I=1..N+1
    for i in 1:Np1
        val = TMP * i
        start[3*i+1] = val     # X(I)
        start[3*i+2] = -val    # Y(I)
        # Z(I) = 0 (already zeroed)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Objective: MG/2*Y(0) + MG*Y(I for I=1..N) + MG/2*Y(N+1)
    # Y(0) = x[2], Y(N+1) = x[3*(N+1)+2] = x[3002]
    ExaModels.@add_obj(c, MG2 * x[2])                 # Y(0)
    ExaModels.@add_obj(c, MG * x[3*i+2] for i in 1:N)  # Y(1)..Y(N)
    ExaModels.@add_obj(c, MG2 * x[3*Np1+2])            # Y(N+1) = x[3002]

    # Constraints C(I) for I=1..N+1:
    # (X(I)-X(I-1))^2 + (Y(I)-Y(I-1))^2 + (Z(I)-Z(I-1))^2 - BLSQ = 0
    # X(I)=x[3I+1], X(I-1)=x[3(I-1)+1]=x[3I-2], etc.
    ExaModels.@add_con(
        c,
        (x[3*i+1] - x[3*i-2])^2 + (x[3*i+2] - x[3*i-1])^2 + (x[3*i+3] - x[3*i])^2 - BLSQ
        for i in 1:Np1
    )

    return ExaModels.ExaModel(c; kwargs...)
end
