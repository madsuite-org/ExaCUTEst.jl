# FMINSRF2: Free boundary minimum surface problem, P=75 default (P^2=5625 variables)
# Variables: X(I,J) for I,J=1..P, all free
# Variable index: (J-1)*P + I
# Objective:
#   sum_{I,J=1}^{P-1} sqrt(0.5*(a(I,J)^2+b(I,J)^2) + 1/(P-1)^2)
#   + X(MID,MID)^2 / P^2
# where a(I,J) = X(I,J)-X(I+1,J+1), b(I,J) = X(I+1,J)-X(I,J+1), MID=floor(P/2)
function fminsrf2_model(P = 75; T = Float64, backend = nothing, kwargs...)
    NV   = P * P
    PM1  = P - 1
    MID  = div(P, 2)
    PARAM  = T(0.5) * T(PM1)^2    # 0.5*(P-1)^2
    SCALE  = T(1) / T(PM1)^2      # 1/(P-1)^2
    P2     = T(P)^2

    # Variable index
    X_idx(i, j) = (j - 1) * P + i

    # Starting point
    H00    = T(1)
    SLOPEJ = T(4)
    SLOPEI = T(8)
    WTOE   = SLOPEJ / T(PM1)   # step in J direction
    STON   = SLOPEI / T(PM1)   # step in I direction
    H01    = H00 + SLOPEJ      # = 5
    H10    = H00 + SLOPEI      # = 9

    x0 = zeros(T, NV)
    # Lower and upper edges (J=1..P, I=1 and I=P)
    for j in 1:P
        x0[X_idx(1, j)] = H00 + T(j - 1) * WTOE    # X(1,J)
        x0[X_idx(P, j)] = H10 + T(j - 1) * WTOE    # X(P,J)
    end
    # Left and right edges (I=2..P-1, J=1 and J=P)
    for i in 2:(P - 1)
        x0[X_idx(i, 1)] = H00 + T(i - 1) * STON    # X(I,1)
        x0[X_idx(i, P)] = H01 + T(i - 1) * STON    # X(I,P)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # S(I,J) groups: SQROOT type with SCALE σ
    # v_g = PARAM*(a^2+b^2) - rhs; rhs = -1 (constant -1)
    # v_g = PARAM*(a^2+b^2) + 1
    # contribution = σ * sqrt(v_g/σ) = sqrt(σ * v_g) = sqrt(SCALE*(PARAM*(a^2+b^2)+1))
    # = sqrt((0.5*(a^2+b^2) + 1/(P-1)^2))... wait:
    # SCALE*v_g = (1/(P-1)^2) * (0.5*(P-1)^2*(a^2+b^2) + 1) = 0.5*(a^2+b^2) + 1/(P-1)^2
    sq_data = [(
        x11 = X_idx(i, j),
        x22 = X_idx(i + 1, j + 1),
        x21 = X_idx(i + 1, j),
        x12 = X_idx(i, j + 1),
    ) for i in 1:PM1 for j in 1:PM1]

    RPM1 = T(PM1)
    ExaModels.@add_obj(c,
        sqrt(T(0.5) * (x[d.x11] - x[d.x22])^2 + T(0.5) * (x[d.x21] - x[d.x12])^2 + SCALE) / RPM1
        for d in sq_data)

    # MID group: L2 type with SCALE P^2 → x(MID,MID)^2/P^2
    ExaModels.@add_obj(c, x[X_idx(MID, MID)]^2 / P2)

    return ExaModels.ExaModel(c; kwargs...)
end
