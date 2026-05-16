# POLYGON: Maximum area polygon problem
# Variables: R(I), THETA(I) for I=1..NV
# R(NV) fixed to 0, THETA(NV) fixed to pi
# R(I) in [0,1], THETA(I) in [0,pi] for I=1..NV-1
# Objective (N type): -0.5 * sum_{I=1}^{NV-1} R(I+1)*R(I)*sin(THETA(I+1)-THETA(I))
# Constraints O(I) for I=1..NV-1 (G type): THETA(I+1) - THETA(I) >= 0
# Constraints D(I,J) for 1<=I<J<=NV (L type):
#   R(I)^2 + R(J)^2 - 2*R(I)*R(J)*cos(THETA(J)-THETA(I)) <= 1
# Variable ordering: R(1)..R(NV), THETA(1)..THETA(NV)
function polygon_model(NV = 100; T = Float64, backend = nothing, kwargs...)
    PI = T(pi)
    nvar = 2 * NV  # R(1..NV), THETA(1..NV)

    idx_R(i) = i
    idx_T(i) = NV + i

    lvar = fill(T(0.0), nvar)
    uvar = fill(T(Inf), nvar)

    # R(I) in [0,1] for I=1..NV-1
    for i in 1:NV-1
        uvar[idx_R(i)] = T(1.0)
        uvar[idx_T(i)] = PI
    end
    # R(NV) = 0 (fixed)
    lvar[idx_R(NV)] = T(0.0); uvar[idx_R(NV)] = T(0.0)
    # THETA(NV) = pi (fixed)
    lvar[idx_T(NV)] = PI; uvar[idx_T(NV)] = PI

    # Starting point: R(I) = (NV+1-I)^2 * 4/(NV+1)^2, THETA(I) = pi*I/NV
    start = zeros(T, nvar)
    for i in 1:NV-1
        RATR = T((NV + 1 - i)^2 * i) * T(4.0) / T((NV + 1)^2)
        RATT = PI * T(i) / T(NV)
        # Actually from SIF: RATR = (NV+1-I)*I/(NV+1)^2*4 ... let me recompute
        # R/ RATR (NV+1)^2 4.0 → RATR = 4/(NV+1)^2
        # R- RATRI NV+1 I → temp = NV+1-I
        # R* RATRI RATRI I → temp = (NV+1-I)*I
        # R* RATRI RATRI RATR → RATRI = (NV+1-I)*I * 4/(NV+1)^2
        RATRI = T((NV + 1 - i) * i) * T(4) / T((NV + 1)^2)
        RATTI = PI * T(i) / T(NV)
        start[idx_R(i)] = RATRI
        start[idx_T(i)] = RATTI
    end
    start[idx_T(NV)] = PI

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Objective: AREA = -0.5 * sum_{I=1}^{NV-1} R(I)*R(I+1)*sin(THETA(I+1)-THETA(I))
    area_data = [(ri=Int32(i), ri1=Int32(i+1), ti=Int32(NV+i), ti1=Int32(NV+i+1)) for i in 1:NV-1]
    ExaModels.@add_obj(c, T(-0.5) * x[d.ri] * x[d.ri1] * sin(x[d.ti1] - x[d.ti]) for d in area_data)

    # Distance constraints D(I,J) for I<J: R(I)^2 + R(J)^2 - 2*R(I)*R(J)*cos(THETA(J)-THETA(I)) - 1 <= 0
    dij_pairs = [(ri=Int32(i), rj=Int32(j), ti=Int32(NV+i), tj=Int32(NV+j)) for i in 1:NV-1 for j in i+1:NV]
    ExaModels.@add_con(c,
        x[d.ri]^2 + x[d.rj]^2 - T(2) * x[d.ri] * x[d.rj] * cos(x[d.tj] - x[d.ti]) - T(1.0)
        for d in dij_pairs;
        lcon = T(-Inf), ucon = T(0.0))

    # Ordering constraints O(I): THETA(I+1) - THETA(I) >= 0
    ord_data = [(ti=Int32(NV+i), ti1=Int32(NV+i+1)) for i in 1:NV-1]
    ExaModels.@add_con(c, x[d.ti1] - x[d.ti] for d in ord_data; lcon = T(0.0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
