# ORTHREGE: Orthogonal regression - fit elliptic helix to 3D points
# Variables: P(1..6) (helix params) + X(I),Y(I),Z(I) for I=1..NPTS
# Objective: sum_I [(X-XD)^2 + (Y-YD)^2 + (Z-ZD)^2]
# Constraints:
#   A(I): X(I) - P1 - P4*cos((Z(I)-P3)/P6) = 0
#   B(I): Y(I) - P2 - P5*cos((Z(I)-P3)/P6) = 0
# Note: P6 >= 0.001
# Data: XD=TP4*cos(THETA), YD=TP5*sin(THETA), ZD=TP6*THETA where THETA=(I-1)*2pi/NPTS
function orthrege_model(NPTS = 2500; T = Float64, backend = nothing, kwargs...)
    TP4 = 1.7; TP5 = 0.8; TP6 = 2.0
    PSEED = 237.1531; PSIZE = 0.2
    PI = pi
    INCR = 2 * PI / NPTS

    XD = Vector{T}(undef, NPTS)
    YD = Vector{T}(undef, NPTS)
    ZD = Vector{T}(undef, NPTS)
    for I in 1:NPTS
        THETA = T(I - 1) * T(INCR)
        ST = sin(THETA); CT = cos(THETA)
        R1 = T(TP4) * CT
        R2 = T(TP5) * ST
        R3 = T(TP6) * THETA
        XSEED = THETA * T(PSEED)
        SSEED = cos(XSEED)
        PERT = T(1.0) + T(PSIZE) * SSEED
        XD[I] = R1 * PERT
        YD[I] = R2 * PERT
        ZD[I] = R3 * PERT
    end

    # Variables: P(1..6), then X(I), Y(I), Z(I) for I=1..NPTS
    nvar = 6 + 3 * NPTS
    idx_P(k) = k
    idx_X(i) = 6 + i
    idx_Y(i) = 6 + NPTS + i
    idx_Z(i) = 6 + 2*NPTS + i

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    lvar[idx_P(6)] = T(0.001)  # P6 >= 0.001

    start = zeros(T, nvar)
    start[idx_P(1)] = T(1.0); start[idx_P(2)] = T(0.0); start[idx_P(3)] = T(1.0)
    start[idx_P(4)] = T(1.0); start[idx_P(5)] = T(0.0); start[idx_P(6)] = T(0.25)
    for I in 1:NPTS
        start[idx_X(I)] = XD[I]; start[idx_Y(I)] = YD[I]; start[idx_Z(I)] = ZD[I]
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, θ_XD, XD[I] for I in 1:NPTS)
    ExaModels.@add_par(c, θ_YD, YD[I] for I in 1:NPTS)
    ExaModels.@add_par(c, θ_ZD, ZD[I] for I in 1:NPTS)

    # Objective
    ExaModels.@add_obj(c, (x[idx_X(I)] - θ_XD[I])^2 for I in 1:NPTS)
    ExaModels.@add_obj(c, (x[idx_Y(I)] - θ_YD[I])^2 for I in 1:NPTS)
    ExaModels.@add_obj(c, (x[idx_Z(I)] - θ_ZD[I])^2 for I in 1:NPTS)

    # Constraint A(I): X(I) - P1 - P4*cos((Z(I)-P3)/P6) = 0
    ExaModels.@add_con(c,
        x[idx_X(I)] - x[idx_P(1)] - x[idx_P(4)] * cos((x[idx_Z(I)] - x[idx_P(3)]) / x[idx_P(6)])
        for I in 1:NPTS)

    # Constraint B(I): Y(I) - P2 - P5*cos((Z(I)-P3)/P6) = 0
    ExaModels.@add_con(c,
        x[idx_Y(I)] - x[idx_P(2)] - x[idx_P(5)] * cos((x[idx_Z(I)] - x[idx_P(3)]) / x[idx_P(6)])
        for I in 1:NPTS)

    return ExaModels.ExaModel(c; kwargs...)
end
