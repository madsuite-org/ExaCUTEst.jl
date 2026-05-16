# ORTHREGC: Orthogonal regression - fit ellipse to perturbed elliptic data
# Variables: H11,H12,H22,G1,G2 (5 params) + X(I),Y(I) for I=1..NPTS
# Constraints E(I): H11*X^2 + 2*H12*X*Y + H22*Y^2 - 2*G1*X - 2*G2*Y = 1
function orthregc_model(NPTS = 2500; T = Float64, backend = nothing, kwargs...)
    V11 = 2.0; V12 = 1.0; V22 = 2.0
    PSEED = 237.1531; PSIZE = 0.2
    PI = pi
    INCR = 2 * PI / NPTS
    C3 = cos(V22); S3 = sin(V22)

    XD = Vector{T}(undef, NPTS)
    YD = Vector{T}(undef, NPTS)
    for I in 1:NPTS
        THETA = T(I - 1) * T(INCR)
        ST = sin(THETA); CT = cos(THETA)
        U1 = V11 * CT; U2 = V12 * ST
        R1 = U1 * C3 + U2 * (-S3)
        R2 = U1 * S3 + U2 * C3
        XSEED = THETA * T(PSEED)
        PERT = T(1.0) + T(PSIZE) * cos(XSEED)
        XD[I] = T(R1) * PERT
        YD[I] = T(R2) * PERT
    end

    nvar = 5 + 2 * NPTS
    idx_H11=1; idx_H12=2; idx_H22=3; idx_G1=4; idx_G2=5
    idx_X(i) = 5 + i
    idx_Y(i) = 5 + NPTS + i

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    start = zeros(T, nvar)
    start[idx_H11] = T(1.0); start[idx_G1] = T(1.0); start[idx_G2] = T(1.0)
    for I in 1:NPTS
        start[idx_X(I)] = XD[I]; start[idx_Y(I)] = YD[I]
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, θ_XD, XD[I] for I in 1:NPTS)
    ExaModels.@add_par(c, θ_YD, YD[I] for I in 1:NPTS)

    ExaModels.@add_obj(c, (x[idx_X(I)] - θ_XD[I])^2 for I in 1:NPTS)
    ExaModels.@add_obj(c, (x[idx_Y(I)] - θ_YD[I])^2 for I in 1:NPTS)

    ExaModels.@add_con(c,
        x[idx_H11] * x[idx_X(I)]^2 +
        T(2) * x[idx_H12] * x[idx_X(I)] * x[idx_Y(I)] +
        x[idx_H22] * x[idx_Y(I)]^2 -
        T(2) * x[idx_G1] * x[idx_X(I)] -
        T(2) * x[idx_G2] * x[idx_Y(I)] - T(1.0)
        for I in 1:NPTS)

    return ExaModels.ExaModel(c; kwargs...)
end
