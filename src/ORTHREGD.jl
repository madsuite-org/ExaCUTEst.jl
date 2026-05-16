# ORTHREGD: Orthogonal regression - fit a planar circle to data points
# Variables: Z1, Z2, Z3 (3 curve params) + X(I), Y(I) for I=1..NPTS
# Objective: sum_I [(X(I)-XD(I))^2 + (Y(I)-YD(I))^2]  (L2 group types)
# Constraints E(I): EA(I) - EB(I) = 0 for I=1..NPTS where
#   EA element: F = ((X-ZA)^2+(Y-ZB)^2)^2  (TA type with DX=X-ZA, DY=Y-ZB)
#   EB element: F = ((X-ZA)^2+(Y-ZB)^2)*(1+ZC^2)^2  (TB type)
# So E(I) = T^2 - T*(1+Z3^2)^2 = T*(T - (1+Z3^2)^2) = 0
# Data: XD(I) = (1+TZ3^2+cos(THETA_I))*cos(THETA_I)*PERT_I
#       YD(I) = (1+TZ3^2+cos(THETA_I))*sin(THETA_I)*PERT_I
function orthregd_model(NPTS = 2500; T = Float64, backend = nothing, kwargs...)
    TZ3 = 1.7
    PSEED = 237.1531
    PSIZE = 0.2
    PI = pi
    INCR = 2 * PI / NPTS
    TZ3SQ = TZ3^2
    one_plus_TZ3SQ = 1.0 + TZ3SQ

    XD = Vector{T}(undef, NPTS)
    YD = Vector{T}(undef, NPTS)
    for I in 1:NPTS
        THETA = T(I - 1) * T(INCR)
        ST = sin(THETA)
        CT = cos(THETA)
        FACT = T(one_plus_TZ3SQ) + CT
        R1 = FACT * CT
        R2 = FACT * ST
        XSEED = THETA * T(PSEED)
        SSEED = cos(XSEED)
        PERT = T(1.0) + T(PSIZE) * SSEED
        XD[I] = R1 * PERT
        YD[I] = R2 * PERT
    end

    # Variables: Z1(1), Z2(2), Z3(3), X(1..NPTS)(4..NPTS+3), Y(1..NPTS)(NPTS+4..2NPTS+3)
    nvar = 3 + 2 * NPTS
    idx_Z1 = 1; idx_Z2 = 2; idx_Z3 = 3
    idx_X(i) = 3 + i
    idx_Y(i) = 3 + NPTS + i

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)

    start = zeros(T, nvar)
    start[idx_Z1] = T(1.0); start[idx_Z2] = T(0.0); start[idx_Z3] = T(1.0)
    for I in 1:NPTS
        start[idx_X(I)] = XD[I]
        start[idx_Y(I)] = YD[I]
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, θ_XD, XD[I] for I in 1:NPTS)
    ExaModels.@add_par(c, θ_YD, YD[I] for I in 1:NPTS)

    # Objective: L2 groups OX(I) and OY(I)
    ExaModels.@add_obj(c, (x[idx_X(I)] - θ_XD[I])^2 for I in 1:NPTS)
    ExaModels.@add_obj(c, (x[idx_Y(I)] - θ_YD[I])^2 for I in 1:NPTS)

    # Constraints E(I): EA(I) - EB(I) = 0
    # EA: T^2 where T = (X-Z1)^2 + (Y-Z2)^2
    # EB: T * (1+Z3^2)^2
    # E(I) = T^2 - T*(1+Z3^2)^2 = T*(T - (1+Z3^2)^2)
    ExaModels.@add_con(c,
        let T_val = (x[idx_X(I)] - x[idx_Z1])^2 + (x[idx_Y(I)] - x[idx_Z2])^2
            T_val^2 - T_val * (T(1) + x[idx_Z3]^2)^2
        end
        for I in 1:NPTS)

    return ExaModels.ExaModel(c; kwargs...)
end
