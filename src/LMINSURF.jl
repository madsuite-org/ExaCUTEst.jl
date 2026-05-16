# LMINSURF: linear minimum surface problem (P=75 default, n=P^2=5625)
# Same structure as MINSURF but with linear (planar) boundary conditions
# Plane: H(I,J) = H00 + SLOPEI*(I-1)/(P-1) + SLOPEJ*(J-1)/(P-1)
# H00=1, SLOPEI=8, SLOPEJ=4
# Boundary variables are fixed (not free)
# Interior variables are free (XR)
# Objective same as MINSURF: sum sqrt(WEIGHT*(a^2+b^2)+1)*SCALE
function lminsurf_model(P = 75; T = Float64, backend = nothing, kwargs...)
    H00 = T(1)
    SLOPEJ = T(4)
    SLOPEI = T(8)
    rp1 = T(P - 1)
    SCALE = rp1^2
    PARAM = T(0.5) * rp1^2
    WEIGHT = PARAM

    # Variable index: X(I,J) = x[(J-1)*P + I]  (SIF: DO J=1..P, DO I=1..P, X X(I,J))
    idx(i, j) = (j - 1) * P + i
    N = P * P

    STON = SLOPEI / rp1  # increment per step in I direction
    WTOE = SLOPEJ / rp1  # increment per step in J direction

    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf), N)
    start_vec = fill(T(0), N)

    # Boundary conditions on lower and upper edges (I=1 and I=P)
    for j in 1:P
        TH = T(j - 1) * WTOE
        TL = TH + H00        # X(1,j) = H00 + (j-1)*WTOE
        TU = TH + H00 + SLOPEI  # X(P,j) = H00 + SLOPEI + (j-1)*WTOE
        k1 = idx(1, j); lvar[k1] = TL; uvar[k1] = TL; start_vec[k1] = TL
        kP = idx(P, j); lvar[kP] = TU; uvar[kP] = TU; start_vec[kP] = TU
    end
    # Boundary conditions on left and right edges (J=1 and J=P)
    for i in 2:P-1
        TV = T(i - 1) * STON
        TR = TV + H00        # X(i,1) = H00 + (i-1)*STON
        TL = TV + H00 + SLOPEJ  # X(i,P) = H00 + SLOPEJ + (i-1)*STON
        k1 = idx(i, 1); lvar[k1] = TR; uvar[k1] = TR; start_vec[k1] = TR
        kP = idx(i, P); lvar[kP] = TL; uvar[kP] = TL; start_vec[kP] = TL
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start_vec)

    # Objective: sum over (i,j)=1..P-1, 1..P-1 of sqrt(WEIGHT*a^2+WEIGHT*b^2+1)*SCALE
    sq_data = [(k1 = idx(i, j), k2 = idx(i+1, j+1), k3 = idx(i+1, j), k4 = idx(i, j+1))
               for i in 1:P-1 for j in 1:P-1]
    ExaModels.@add_obj(c,
        sqrt(WEIGHT * (x[d.k1] - x[d.k2])^2 + WEIGHT * (x[d.k3] - x[d.k4])^2 + T(1)) * SCALE
        for d in sq_data)

    return ExaModels.ExaModel(c; kwargs...)
end
