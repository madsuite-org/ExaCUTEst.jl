# MINSURF: minimum surface problem (P=7 default, n=(P+1)^2=64)
# Variables: X(i,j) for i=1..P+1, j=1..P+1
# Group type SQROOT: phi(alpha) = sqrt(alpha), with constant RHS -1.0 and scale RPSQ=(P)^2
# So contribution per group = sqrt(RPSQ * (A(i,j)*WEIGHT + B(i,j)*WEIGHT)) / RPSQ
# Wait: group value v = PARAM*A + PARAM*B - 1.0, where PARAM = 0.5*P^2/(P-1)^2...
# Actually: SCALE = RPSQ = (P-1)^2, constant = -1,
# Group S(i,j): v = PARAM*(X(i,j)-X(i+1,j+1))^2 + PARAM*(X(i+1,j)-X(i,j+1))^2 - 1.0
# WEIGHT = PARAM = 0.5*P^2/(P-1)^2... let me recalculate
# PARAM = SQP-1 * 0.5 = (P-1)^2 * 0.5 = 0.5*(P-1)^2
# Elements A(i,j) = (X(i,j)-X(i+1,j+1))^2, B(i,j) = (X(i+1,j)-X(i,j+1))^2
# WEIGHT (from GROUP USES) = RM WEIGHT RPSQ 0.5 = 0.5*(P)^2 ... wait RPSQ = INVP-1^2 = 1/(P-1)^2
# Correction: RP-1 = P-1 as real, INVP-1 = 1/(P-1), RP-1SQ = 1/(P-1)^2, SCALE = 1/RP-1SQ = (P-1)^2
# PARAM = SQP-1 * 0.5 = (P-1)^2 * 0.5
# GROUP S(i,j): linear = none, elements with coeff PARAM
# constant = -1
# SCALE = (P-1)^2
# GROUP TYPE SQROOT: F = sqrt(alpha), G = 0.5/sqrt(alpha), H = -0.25/(alpha*sqrt(alpha))
# Full contribution = SCALE * phi(v/SCALE) = (P-1)^2 * sqrt(v/(P-1)^2) = sqrt(v*(P-1)^2)
# Actually: F(v_g/sigma) * sigma where sigma=SCALE=(P-1)^2:
#   = (P-1)^2 * sqrt(v_g/(P-1)^2) = sqrt(v_g * (P-1)^2) = (P-1)*sqrt(v_g)
# where v_g = PARAM*(a^2 + b^2) - 1 = 0.5*(P-1)^2*(a^2+b^2) - 1
# Hmm, v_g has RHS -1.0 subtracted: v_g = 0.5*(P-1)^2*(a^2+b^2) - 1
# objective = sum sqrt(v_g_raw / sigma) * sigma = sum sqrt(v_g_raw * sigma)
# where sigma = (P-1)^2 and v_g_raw = 0.5*(P-1)^2*(a^2+b^2) - (-1) = 0.5*(P-1)^2*(a^2+b^2) + 1
# Wait CONSTANTS: X MINSURF S(i,j) -1.0, which means RHS = -1.0
# So v_g = elem_contributions - (-1.0)? No: v_g = linear_part + elem_contributions - rhs
# rhs = +1 from the CONSTANTS entry. Contribution = PARAM*(a^2+b^2)
# So v_g = PARAM*(a^2+b^2) - 1 = 0.5*(P-1)^2*(a^2+b^2) - 1
# But CONSTANTS gives RHS= -1.0, meaning v_g = linear + elem - (-1.0) = elem + 1
# = PARAM*(a^2+b^2) + 1 = 0.5*(P-1)^2*(a^2+b^2) + 1
# obj contribution = SCALE * sqrt(v_g/SCALE) = sqrt(v_g * SCALE)
# = sqrt((0.5*(P-1)^2*(a^2+b^2)+1) * (P-1)^2)
# Boundary: X(1,j)=X(P+1,j)=X(i,1)=X(i,P+1)=1 (fixed from BOUNDS: XX=1.0)
# Interior variables free (XR)
# Start: boundary=1.0, interior=0.0
function minsurf_model(P = 7; T = Float64, backend = nothing, kwargs...)
    P1 = P + 1
    RP_1 = T(P)  # RP = P (as real)
    RPSQ = RP_1 * RP_1  # SQP of RP-1 in SIF? Actually let me re-read:
    # RI RP-1 P-1: so RP-1 = P-1 as real
    # INVP-1 = 1/(P-1), RP-1SQ = 1/(P-1)^2, SCALE = 1/RP-1SQ = (P-1)^2
    # SQP-1 = RP-1 * RP-1 = (P-1)^2, PARAM = SQP-1 * 0.5 = 0.5*(P-1)^2
    rp1 = T(P - 1)
    SCALE = rp1^2
    PARAM = T(0.5) * rp1^2
    WEIGHT = PARAM  # WEIGHT = RM WEIGHT RPSQ 0.5, RPSQ = SQP-1 = (P-1)^2, so WEIGHT = 0.5*(P-1)^2 = PARAM

    # Variable index: X(i,j) = x[(j-1)*P1 + i]
    idx(i, j) = (j - 1) * P1 + i
    N = P1 * P1

    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf), N)
    start_vec = fill(T(0), N)

    # Fix boundary to 1.0
    for j in 1:P1
        for i in [1, P1]
            k = idx(i, j); lvar[k] = T(1); uvar[k] = T(1); start_vec[k] = T(1)
        end
    end
    for i in 2:P
        for j in [1, P1]
            k = idx(i, j); lvar[k] = T(1); uvar[k] = T(1); start_vec[k] = T(1)
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start_vec)

    # Objective: sum over (i,j) = 1..P, 1..P of sqrt(WEIGHT*a^2 + WEIGHT*b^2 + 1)*SCALE
    # where a = X(i,j) - X(i+1,j+1), b = X(i+1,j) - X(i,j+1)
    # Note: SCALE * sqrt(v/SCALE) = sqrt(v * SCALE) where v = PARAM*(a^2+b^2) + 1
    sq_data = [(k1 = idx(i, j), k2 = idx(i+1, j+1), k3 = idx(i+1, j), k4 = idx(i, j+1))
               for i in 1:P for j in 1:P]
    ExaModels.@add_obj(c,
        sqrt(WEIGHT * (x[d.k1] - x[d.k2])^2 + WEIGHT * (x[d.k3] - x[d.k4])^2 + T(1)) * SCALE
        for d in sq_data)

    return ExaModels.ExaModel(c; kwargs...)
end
