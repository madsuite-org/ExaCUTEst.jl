# ORTHRGDM: Orthogonal circle fit, modified start to avoid singularity
# Same structure as ORTHREGD but FACT = 1+TZ3SQ + 0.25*CT, NPTS=5000
function orthrgdm_model(NPTS = 5000; T = Float64, backend = nothing, kwargs...)
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
        FACT = T(one_plus_TZ3SQ) + T(0.25) * CT
        R1 = FACT * CT
        R2 = FACT * ST
        XSEED = THETA * T(PSEED)
        SSEED = cos(XSEED)
        PERT = T(1.0) + T(PSIZE) * SSEED
        XD[I] = R1 * PERT
        YD[I] = R2 * PERT
    end

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

    ExaModels.@add_obj(c, (x[idx_X(I)] - θ_XD[I])^2 for I in 1:NPTS)
    ExaModels.@add_obj(c, (x[idx_Y(I)] - θ_YD[I])^2 for I in 1:NPTS)

    ExaModels.@add_con(c,
        let T_val = (x[idx_X(I)] - x[idx_Z1])^2 + (x[idx_Y(I)] - x[idx_Z2])^2
            T_val^2 - T_val * (T(1) + x[idx_Z3]^2)^2
        end
        for I in 1:NPTS)

    return ExaModels.ExaModel(c; kwargs...)
end
