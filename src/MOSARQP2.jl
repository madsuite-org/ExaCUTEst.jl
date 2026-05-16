# MOSARQP2: Convex QP with Laplacian constraints (a third active at solution)
# QLR2-AN-V-V: N=2500 variables, M=700 inequality (G type) constraints
# Source: Morales-Perez & Sargent, 1991. SIF input: Ph. Toint, August 1993.
# Same as MOSARQP1 but quadratic center XC uses 1/3 active pattern (triplet -1,-1,+1)
function mosarqp2_model(; T = Float64, backend = nothing, kwargs...)
    N = 2500
    M = 700
    COND = T(3.0)
    RTN = 50  # floor(sqrt(2500 + 0.1)) = 50

    NNZ = 10
    Y_vals = T[-0.3569732, 0.9871576, 0.5619363, -0.1984624, 0.4653328,
                0.7364367, -0.4560378, -0.6457813, -0.0601357, 0.1035624]
    NZ_fracs = T[0.68971452, 0.13452678, 0.51234678, 0.76591423, 0.20857854,
                  0.85672348, 0.04356789, 0.44692743, 0.30136413, 0.91367489]

    K = Int32[floor(Int32, NZ_fracs[i] * N + T(1.1)) for i in 1:NNZ]
    D = [exp(T(i - 1) / T(N - 1) * COND) for i in 1:N]
    DY = [Y_vals[i] * D[K[i]] for i in 1:NNZ]

    YN2 = sum(y^2 for y in Y_vals)
    neg2_yn2 = T(-2.0) / YN2
    pos4_yn4 = T(4.0) / YN2^2

    # Quadratic center XC: triplet pattern -1,-1,+1 (for 1/3 active)
    XC = zeros(T, N)
    i = 1
    while i <= N - 2
        XC[i] = T(-1.0)
        XC[i+1] = T(-1.0)
        XC[i+2] = T(1.0)
        i += 3
    end
    if i == N - 1
        XC[N-1] = T(-1.0)
        XC[N] = T(1.0)
    elseif i == N
        # N not divisible by 3 with i==N left: handled by the loop ending
    end
    # SIF: DO I=1,N-2 DI 3 (step 3): XC(I)=-1, XC(I+1)=-1, XC(I+2)=1
    # Then AE XC(N-1) -1, AE XC(N) 1
    # Redo this exactly:
    fill!(XC, T(0.0))
    ii = 1
    while ii <= N - 2
        XC[ii] = T(-1.0)
        XC[ii+1] = T(-1.0)
        XC[ii+2] = T(1.0)
        ii += 3
    end
    XC[N-1] = T(-1.0)
    XC[N] = T(1.0)

    YDY = sum(DY[i] * Y_vals[i] for i in 1:NNZ)
    YXC = sum(Y_vals[i] * XC[K[i]] for i in 1:NNZ)
    YDXC = sum(DY[i] * XC[K[i]] for i in 1:NNZ)

    AA = neg2_yn2 * YXC
    DD = pos4_yn4 * YDY
    BB_CC = DD * YXC + neg2_yn2 * YDXC
    DD_half = DD * T(0.5)

    C_grad = [D[i] * XC[i] for i in 1:N]
    for i in 1:NNZ
        ki = K[i]
        C_grad[ki] += DY[i] * AA + Y_vals[i] * BB_CC
    end

    H_diag = copy(D)
    for i in 1:NNZ
        ki = K[i]
        wii = DY[i] * Y_vals[i] * neg2_yn2 + Y_vals[i]^2 * DD_half
        H_diag[ki] += 2 * wii
    end

    off_pairs = [(i = Int32(K[i]), j = Int32(K[jj]),
                  w = T(DY[i]*Y_vals[jj]*neg2_yn2 + DY[jj]*Y_vals[i]*neg2_yn2 + Y_vals[i]*Y_vals[jj]*DD))
                 for i in 1:NNZ for jj in 1:i-1]

    # Build 5-point Laplacian constraints (same structure as MOSARQP1)
    con_data = [(ci = Int32(idx),
                  ali = Int32(mod(idx-1, RTN) >= 1 ? idx-1 : 1),
                  ari = Int32(mod(idx-1, RTN) < RTN-1 ? idx+1 : 1),
                  aui = Int32(div(idx-1, RTN) >= 1 ? idx-RTN : 1),
                  adi = Int32(div(idx-1, RTN) < RTN-1 ? idx+RTN : 1),
                  cl  = T(mod(idx-1, RTN) >= 1 ? -1.0 : 0.0),
                  cr  = T(mod(idx-1, RTN) < RTN-1 ? -1.0 : 0.0),
                  cu  = T(div(idx-1, RTN) >= 1 ? -1.0 : 0.0),
                  cd  = T(div(idx-1, RTN) < RTN-1 ? -1.0 : 0.0),
                  rhs = let have_l = mod(idx-1, RTN) >= 1, have_r = mod(idx-1, RTN) < RTN-1,
                            have_u = div(idx-1, RTN) >= 1, have_d = div(idx-1, RTN) < RTN-1
                            T(3 - Int(have_l) - Int(have_r) - Int(have_u) - Int(have_d)) * T(0.5)
                        end)
                 for idx in 1:M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = zeros(T, N), uvar = fill(T(Inf), N), start = fill(T(0.5), N))

    hd_data = [(i = Int32(i), d = T(H_diag[i])) for i in 1:N]
    ExaModels.@add_obj(c, T(0.5) * d.d * x[d.i]^2 for d in hd_data)

    cg_data = [(i = Int32(i), g = T(C_grad[i])) for i in 1:N]
    ExaModels.@add_obj(c, d.g * x[d.i] for d in cg_data)

    if !isempty(off_pairs)
        ExaModels.@add_obj(c, d.w * x[d.i] * x[d.j] for d in off_pairs)
    end

    ExaModels.@add_con(c,
        T(4) * x[d.ci] + d.cl * x[d.ali] + d.cr * x[d.ari] + d.cu * x[d.aui] + d.cd * x[d.adi] - d.rhs
        for d in con_data;
        lcon = T(0.0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
