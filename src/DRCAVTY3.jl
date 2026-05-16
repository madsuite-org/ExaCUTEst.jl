# DRCAVTY3: Like DRCAVTY1 but RE=4500
function drcavty3_model(M = 63; T = Float64, backend = nothing, kwargs...)
    RE = T(4500.0)
    RM2 = M + 2
    H = T(1) / T(RM2)
    NI = M + 4
    NV = NI * NI
    RE4 = RE / 4

    y_idx(i, j) = (i + 1) * NI + (j + 1) + 1

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    start = zeros(T, NV)

    for j in -1:M+2
        lvar[y_idx(-1, j)] = T(0); uvar[y_idx(-1, j)] = T(0)
        lvar[y_idx(0, j)]  = T(0); uvar[y_idx(0, j)]  = T(0)
    end
    for i in 1:M
        lvar[y_idx(i, -1)]   = T(0); uvar[y_idx(i, -1)]   = T(0)
        lvar[y_idx(i, 0)]    = T(0); uvar[y_idx(i, 0)]    = T(0)
        lvar[y_idx(i, M+1)]  = T(0); uvar[y_idx(i, M+1)]  = T(0)
        lvar[y_idx(i, M+2)]  = T(0); uvar[y_idx(i, M+2)]  = T(0)
    end
    for j in -1:M+2
        lvar[y_idx(M+1, j)] = -H/2; uvar[y_idx(M+1, j)] = -H/2; start[y_idx(M+1, j)] = -H/2
        lvar[y_idx(M+2, j)] =  H/2; uvar[y_idx(M+2, j)] =  H/2; start[y_idx(M+2, j)] =  H/2
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = start)

    inner = [(y_ij     = y_idx(i, j),
              y_im1j   = y_idx(i-1, j), y_ip1j   = y_idx(i+1, j),
              y_ijm1   = y_idx(i, j-1), y_ijp1   = y_idx(i, j+1),
              y_im1jm1 = y_idx(i-1, j-1), y_im1jp1 = y_idx(i-1, j+1),
              y_ip1jm1 = y_idx(i+1, j-1), y_ip1jp1 = y_idx(i+1, j+1),
              y_im2j   = y_idx(i-2, j), y_ip2j   = y_idx(i+2, j),
              y_ijm2   = y_idx(i, j-2), y_ijp2   = y_idx(i, j+2))
             for i in 1:M for j in 1:M]

    ExaModels.@add_con(c,
        T(20) * x[d.y_ij] +
        T(-8) * (x[d.y_im1j] + x[d.y_ip1j] + x[d.y_ijm1] + x[d.y_ijp1]) +
        T(2) * (x[d.y_im1jp1] + x[d.y_ip1jm1] + x[d.y_im1jm1] + x[d.y_ip1jp1]) +
        x[d.y_im2j] + x[d.y_ip2j] + x[d.y_ijm2] + x[d.y_ijp2] +
        RE4 * (x[d.y_ijp1] - x[d.y_ijm1]) *
              (x[d.y_im2j] + x[d.y_im1jm1] + x[d.y_im1jp1] -
               T(4) * x[d.y_im1j] + T(4) * x[d.y_ip1j] -
               x[d.y_ip1jm1] - x[d.y_ip1jp1] - x[d.y_ip2j]) -
        RE4 * (x[d.y_ip1j] - x[d.y_im1j]) *
              (x[d.y_ijm2] + x[d.y_im1jm1] + x[d.y_ip1jm1] -
               T(4) * x[d.y_ijm1] + T(4) * x[d.y_ijp1] -
               x[d.y_im1jp1] - x[d.y_ip1jp1] - x[d.y_ijp2])
        for d in inner)

    return ExaModels.ExaModel(c; kwargs...)
end
