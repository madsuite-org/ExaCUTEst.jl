# ORTHREGF: Orthogonal regression - fit torus to 3D points
# Variables: P(1..5) (torus params) + X(I,J),Y(I,J),Z(I,J) for I,J=1..NPTS
# where NPTS^2 data points
# Constraint A(I,J): (sqrt((X-P1)^2+(Y-P2)^2)/P4 - 1)^2 + (Z-P3)^2 - P5^2 = 0
# Bounds: P4 >= 0.001, P5 >= 0.001
# Torus: center (P1,P2,P3), major radius P4, minor radius P5
function orthregf_model(NPTS = 40; T = Float64, backend = nothing, kwargs...)
    TP4 = 1.7; TP5 = 0.8
    PSEED = 237.1531; PSIZE = 0.2
    PI = pi
    INCR = 2 * PI / NPTS

    # Generate NPTS*NPTS data points
    NP = NPTS * NPTS
    XD = Vector{T}(undef, NP)
    YD = Vector{T}(undef, NP)
    ZD = Vector{T}(undef, NP)
    for I in 1:NPTS
        THETA1 = T(I - 1) * T(INCR)
        ST1 = sin(THETA1); CT1 = cos(THETA1)
        P5CT1 = T(TP5) * CT1
        P4P5CT1 = T(TP4) + P5CT1
        R3 = T(TP5) * ST1
        for J in 1:NPTS
            THETA2 = T(J - 1) * T(INCR)
            ST2 = sin(THETA2); CT2 = cos(THETA2)
            R1 = P4P5CT1 * CT2
            R2 = P4P5CT1 * ST2
            XSEED = THETA2 * T(PSEED)
            SSEED = cos(XSEED)
            PERT = T(1.0) + T(PSIZE) * SSEED
            idx = (I - 1) * NPTS + J
            XD[idx] = R1 * PERT
            YD[idx] = R2 * PERT
            ZD[idx] = R3 * PERT
        end
    end

    # Variables: P(1..5), X(1..NP), Y(1..NP), Z(1..NP)
    nvar = 5 + 3 * NP
    idx_P(k) = k
    idx_X(i) = 5 + i
    idx_Y(i) = 5 + NP + i
    idx_Z(i) = 5 + 2*NP + i

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    lvar[idx_P(4)] = T(0.001)
    lvar[idx_P(5)] = T(0.001)

    start = zeros(T, nvar)
    start[idx_P(1)] = T(1.0); start[idx_P(2)] = T(0.0); start[idx_P(3)] = T(1.0)
    start[idx_P(4)] = T(1.0); start[idx_P(5)] = T(0.5)
    for i in 1:NP
        start[idx_X(i)] = XD[i]; start[idx_Y(i)] = YD[i]; start[idx_Z(i)] = ZD[i]
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, θ_XD, XD[i] for i in 1:NP)
    ExaModels.@add_par(c, θ_YD, YD[i] for i in 1:NP)
    ExaModels.@add_par(c, θ_ZD, ZD[i] for i in 1:NP)

    # Objective: L2 groups
    ExaModels.@add_obj(c, (x[idx_X(i)] - θ_XD[i])^2 for i in 1:NP)
    ExaModels.@add_obj(c, (x[idx_Y(i)] - θ_YD[i])^2 for i in 1:NP)
    ExaModels.@add_obj(c, (x[idx_Z(i)] - θ_ZD[i])^2 for i in 1:NP)

    # Constraint A(I,J): (sqrt((X-P1)^2+(Y-P2)^2)/P4 - 1)^2 + (Z-P3)^2 - P5^2 = 0
    ExaModels.@add_con(c,
        (sqrt((x[idx_X(i)] - x[idx_P(1)])^2 + (x[idx_Y(i)] - x[idx_P(2)])^2) / x[idx_P(4)] - T(1))^2 +
        (x[idx_Z(i)] - x[idx_P(3)])^2 - x[idx_P(5)]^2
        for i in 1:NP)

    return ExaModels.ExaModel(c; kwargs...)
end
