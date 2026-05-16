# RAYBENDL: Ray bending problem
# 2*(NKNOTS+1) variables: X(0..NKNOTS), Z(0..NKNOTS)
# Fixed: X(0)=XSRC=0, Z(0)=ZSRC=0, X(NKNOTS)=XRCV=100, Z(NKNOTS)=ZRCV=100
# Objective: sum_I TIME(I) for I=1..NKNOTS with SCALE=2
#   TIME(I) uses element TT(X(I-1),X(I),Z(I-1),Z(I))
#   TT element:
#     IV: DX = X2 - X1, ZZ0 = Z1, ZZ1 = Z2
#     CZ = 0.01, C0 = 1 + CZ*ZZ0, C1 = 1 + CZ*ZZ1
#     V = 1/C1 + 1/C0
#     DZ1 = ZZ1 - ZZ0
#     R = sqrt(DX^2 + DZ1^2)
#     F = V * R
#   SCALE=2: contribution = V*R / 2
# Variable ordering: X(0..NKNOTS), Z(0..NKNOTS) — total 2*(NKNOTS+1)
function raybendl_model(NKNOTS = 1024; T = Float64, backend = nothing, kwargs...)
    CZ = T(0.01)
    XSRC = T(0.0); ZSRC = T(0.0)
    XRCV = T(100.0); ZRCV = T(100.0)

    nvar = 2 * (NKNOTS + 1)
    idx_X(i) = i + 1       # X(0..NKNOTS) → positions 1..NKNOTS+1
    idx_Z(i) = NKNOTS + 2 + i  # Z(0..NKNOTS) → positions NKNOTS+2..2*(NKNOTS+1)

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)

    # Fix endpoints
    lvar[idx_X(0)] = XSRC; uvar[idx_X(0)] = XSRC
    lvar[idx_Z(0)] = ZSRC; uvar[idx_Z(0)] = ZSRC
    lvar[idx_X(NKNOTS)] = XRCV; uvar[idx_X(NKNOTS)] = XRCV
    lvar[idx_Z(NKNOTS)] = ZRCV; uvar[idx_Z(NKNOTS)] = ZRCV

    # Starting point: equidistant on straight line
    XRANGE = XRCV - XSRC
    ZRANGE = ZRCV - ZSRC
    start = zeros(T, nvar)
    for i in 0:NKNOTS
        frac = T(i) / T(NKNOTS)
        start[idx_X(i)] = XSRC + frac * XRANGE
        start[idx_Z(i)] = ZSRC + frac * ZRANGE
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Objective: sum_I (1/2) * TT(X(I-1),X(I),Z(I-1),Z(I)) for I=1..NKNOTS
    # TT element: F = V*R where
    #   DX = X(I) - X(I-1), DZ1 = Z(I) - Z(I-1)
    #   C0 = 1 + CZ*Z(I-1), C1 = 1 + CZ*Z(I)
    #   V = 1/C1 + 1/C0
    #   R = sqrt(DX^2 + DZ1^2)
    # SCALE=2: contribution = (V*R)/2
    ExaModels.@add_obj(c,
        T(0.5) * (T(1) / (T(1) + CZ * x[idx_Z(i)]) + T(1) / (T(1) + CZ * x[idx_Z(i-1)])) *
        sqrt((x[idx_X(i)] - x[idx_X(i-1)])^2 + (x[idx_Z(i)] - x[idx_Z(i-1)])^2)
        for i in 1:NKNOTS)

    return ExaModels.ExaModel(c; kwargs...)
end
