# NLMSURF: minimum surface with nonlinear boundary conditions (P=75 default)
# Same objective as MINSURF/LMINSURF but with nonlinear boundary values
# Boundary (t = (j-1)/(P-1)):
#   X(1,J) = 1 + 4t + 10(1+t)^2 = LOWER (lower edge)  [I=1 in SIF = i=1]
#   X(P,J) = 9 + 4t + 10t^2 = UPPER (upper edge)  [I=P]
# Boundary (t = (i-1)/(P-1)):
#   X(I,1) = 1 + 8t + 10(1-t)^2 = RIGHT (left side, J=1)
#   X(I,P) = 5 + 8t + 10(2-t)^2 = LEFT (right side, J=P)
# Note: X(I,J) uses SIF ordering: DO J 1..P, DO I 1..P, so x[(J-1)*P+I]
function nlmsurf_model(P = 75; T = Float64, backend = nothing, kwargs...)
    rp1 = T(P - 1)
    SCALE = rp1^2
    PARAM = T(0.5) * rp1^2
    WEIGHT = PARAM
    INVP1 = T(1) / rp1

    idx(i, j) = (j - 1) * P + i
    N = P * P

    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf), N)
    start_vec = fill(T(0), N)

    # Compute boundary values
    # X(1,J) and X(P,J): t = (J-1)/(P-1)
    for j in 1:P
        t = T(j - 1) * INVP1
        t2 = t * t
        lower = T(10) * (T(1) + t)^2 + T(4) * t + T(1)  # LOWER
        upper = T(10) * t2 + T(4) * t + T(9)  # UPPER
        k1 = idx(1, j); lvar[k1] = lower; uvar[k1] = lower; start_vec[k1] = lower
        kP = idx(P, j); lvar[kP] = upper; uvar[kP] = upper; start_vec[kP] = upper
    end
    # X(I,1) and X(I,P): t = (I-1)/(P-1)
    for i in 2:P-1
        t = T(i - 1) * INVP1
        right = T(10) * (T(1) - t)^2 + T(8) * t + T(1)  # RIGHT (J=1)
        left  = T(10) * (T(2) - t)^2 + T(8) * t + T(5)  # LEFT (J=P)
        k1 = idx(i, 1); lvar[k1] = right; uvar[k1] = right; start_vec[k1] = right
        kP = idx(i, P); lvar[kP] = left;  uvar[kP] = left;  start_vec[kP] = left
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start_vec)

    sq_data = [(k1 = idx(i, j), k2 = idx(i+1, j+1), k3 = idx(i+1, j), k4 = idx(i, j+1))
               for i in 1:P-1 for j in 1:P-1]
    ExaModels.@add_obj(c,
        sqrt(WEIGHT * (x[d.k1] - x[d.k2])^2 + WEIGHT * (x[d.k3] - x[d.k4])^2 + T(1)) * SCALE
        for d in sq_data)

    return ExaModels.ExaModel(c; kwargs...)
end
