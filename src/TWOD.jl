function twod_model(N = 40; T = Float64, backend = nothing, kwargs...)
    # TWOD: parabolic PDE optimal control problem
    # Y(K,I,J): K=0..M, I=0..N, J=0..N  [state variables]
    # U(K,J):   K=1..M, J=0..N-1         [control variables]
    # N and M are equal (M=N)
    # Variables: (N+1)^2*(M+1) + M*(N) total

    M  = N
    N1 = N - 1
    N2 = N - 2
    M1 = M - 1

    A    = 0.001
    UA   = 2.0
    ONE  = 1.0
    HALF = 0.5

    DX   = ONE / N
    DY   = ONE / M
    DT   = ONE / M

    H2   = DX * DX
    DXDY = DX * DY
    DTDX = DT * DX
    ADTDX = A * DTDX

    c1 = ONE / DT + 2 / H2        # 1/DT + 2/H2
    cm1 = -ONE / DT + 2 / H2      # -1/DT + 2/H2
    chalf = -HALF / (2 * H2)      # -.1/(2*H2) (note: factor .1 = A*DX, but looking at GROUPS: -.1/2H2)
    # Actually the coefficient name -.1/2H2 in SIF means the parameter is (-.1)/(2*H2)
    # where .1 = A = 0.001? No: RE -.1/2H2   -HALF     H2 means -0.5/H2
    # Let me recheck: R/ -.1/2H2   -HALF    H2 → -.1/2H2 = -0.5/H2
    # But the label says "-.1" which is just the name, not 0.1
    coeff_nb = -HALF / H2         # = -.5/H2 = -.1/2H2 label in SIF

    # 1/2DY, -2/DY, 3/2DY+1
    c_1per2DY = ONE / (2 * DY)
    cm2perDY  = -2 / DY
    c3per2DYp1 = 3 / (2 * DY) + ONE
    c_1per2DX = ONE / (2 * DX)
    cm2perDX  = -2 / DX
    c3per2DXp1 = 3 / (2 * DX) + ONE

    # Variable indexing:
    # Y(K,I,J): K=0..M, I=0..N, J=0..N
    NP1 = N + 1
    MP1 = M + 1
    # y_idx(K,I,J) = K * NP1*NP1 + I * NP1 + J + 1
    y_idx(K, I, J) = K * NP1 * NP1 + I * NP1 + J + 1
    NY = MP1 * NP1 * NP1   # total Y variables

    # U(K,J): K=1..M, J=0..N-1
    # u_idx(K,J) = NY + (K-1)*N + J + 1
    u_idx(K, J) = NY + (K - 1) * N + J + 1
    NU = M * N

    NV = NY + NU

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    x0   = zeros(T, NV)

    # Y(0,I,J) = 0 for all I,J
    for I in 0:N, J in 0:N
        vi = y_idx(0, I, J)
        lvar[vi] = T(0); uvar[vi] = T(0)
    end

    # Y(K,I,J) ∈ [0, 0.8] for K=1..M, I=0..N, J=0..N
    for K in 1:M, I in 0:N, J in 0:N
        vi = y_idx(K, I, J)
        lvar[vi] = T(0); uvar[vi] = T(0.8)
    end

    # U(K,J) ∈ [0, UA=2.0] for K=1..M, J=0..N-1
    for K in 1:M, J in 0:N-1
        vi = u_idx(K, J)
        lvar[vi] = T(0); uvar[vi] = T(UA); x0[vi] = T(UA)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective:
    # SQD element at (M,I,J): F = (Y(M,I,J) - YP)^2 where YP = 0.5*DX*DY*I*J + 0.25
    # SQ element at (K,I): F = U(K,I)^2
    # Weights follow the SIF GROUP USES section (note: SIF has a bug where edge J=N is
    # replaced by N-1 copies of corner (N,N) in the DO J loop)

    # Corner (0,0): YP=0.25, coeff=.125DXDY
    ExaModels.@add_obj(c, T(0.125*DXDY) * (x[y_idx(M, 0, 0)] - T(0.25))^2)
    # Corner (0,N): YP=0.25 (I=0), coeff=.125DXDY
    ExaModels.@add_obj(c, T(0.125*DXDY) * (x[y_idx(M, 0, N)] - T(0.25))^2)
    # Corner (N,0): YP=0.25 (J=0), coeff=.125DXDY
    ExaModels.@add_obj(c, T(0.125*DXDY) * (x[y_idx(M, N, 0)] - T(0.25))^2)
    # Corner (N,N): YP=0.5*DX*DY*N*N+0.25=0.75, coeff = .125DXDY + (N-1)*.25DXDY (SIF bug)
    let coeff_nn = T((0.125 + (N-1)*0.25) * DXDY), yp_nn = T(0.5 * DX * DY * N * N + 0.25)
        ExaModels.@add_obj(c, coeff_nn * (x[y_idx(M, N, N)] - yp_nn)^2)
    end

    # Edge I=0, J=1..N-1 and I=N, J=1..N-1: YP = 0.5*DX*DY*I*j+0.25, coeff=.25DXDY
    # I=0 → YP=0.25, I=N → YP=0.5*DX*DY*N*j+0.25 = j/(2N)+0.25
    let DXDYq = T(0.25 * DXDY)
        yp_Nj = T[0.5 * DX * DY * N * j + 0.25 for j in 1:N1]
        ExaModels.@add_par(c, yp_Nj_par, yp_Nj)
        ExaModels.@add_obj(c,
            DXDYq * (x[y_idx(M, 0, j)] - T(0.25))^2 for j in 1:N1)
        ExaModels.@add_obj(c,
            DXDYq * (x[y_idx(M, N, j)] - yp_Nj_par[j])^2 for j in 1:N1)
        # Edge J=0, I=1..N-1: YP=0.25 (J=0)
        ExaModels.@add_obj(c,
            DXDYq * (x[y_idx(M, i, 0)] - T(0.25))^2 for i in 1:N1)
        # Edge J=N, I=1..N-1: NOT added (SIF bug uses E(M,N,N) instead)
    end

    # SQD interior terms: I=1..N-1, J=1..N-1
    let DXDYhalf = T(0.5 * DXDY)
        yp_ij = T[0.5 * DX * DY * i * j + 0.25 for j in 1:N1 for i in 1:N1]
        ExaModels.@add_par(c, yp_ij_par, yp_ij)
        ExaModels.@add_obj(c,
            DXDYhalf * (x[y_idx(M, i, j)] - yp_ij_par[(j-1)*N1 + i])^2
            for j in 1:N1 for i in 1:N1)
    end

    # SQ control terms: K=1..M-1
    let ADTDXh = T(0.5 * ADTDX)
        ExaModels.@add_obj(c, ADTDXh * x[u_idx(K, I)]^2 for K in 1:M1 for I in 1:N1)
    end
    # Last time step K=M
    let ADTDXq = T(0.25 * ADTDX)
        ExaModels.@add_obj(c, ADTDXq * x[u_idx(M, I)]^2 for I in 1:N1)
    end

    # Constraints P(K,I,J): K=0..M-1, I=1..N-1, J=1..N-1 (equality, =0)
    # P(K,I,J): (1/DT+2/H2)*Y(K+1,I,J) + (-1/DT+2/H2)*Y(K,I,J)
    #           + (-.5/H2)*(Y(K,I,J-1)+Y(K,I,J+1)+Y(K,I-1,J)+Y(K,I+1,J))
    #           + (-.5/H2)*(Y(K+1,I-1,J)+Y(K+1,I+1,J)+Y(K+1,I,J-1)+Y(K+1,I,J+1)) = 0
    let c1T = T(c1), cm1T = T(cm1), cnbT = T(coeff_nb)
        pcon_data = [(ykp1ij  = y_idx(K+1, I, J),
                      ykij    = y_idx(K, I, J),
                      ykijm1  = y_idx(K, I, J-1),
                      ykijp1  = y_idx(K, I, J+1),
                      ykim1j  = y_idx(K, I-1, J),
                      ykip1j  = y_idx(K, I+1, J),
                      ykp1im1j = y_idx(K+1, I-1, J),
                      ykp1ip1j = y_idx(K+1, I+1, J),
                      ykp1ijm1 = y_idx(K+1, I, J-1),
                      ykp1ijp1 = y_idx(K+1, I, J+1))
                     for K in 0:M1 for I in 1:N1 for J in 1:N1]
        ExaModels.@add_con(c,
            c1T  * x[d.ykp1ij]
            + cm1T * x[d.ykij]
            + cnbT * (x[d.ykijm1] + x[d.ykijp1] + x[d.ykim1j] + x[d.ykip1j]
                    + x[d.ykp1im1j] + x[d.ykp1ip1j] + x[d.ykp1ijm1] + x[d.ykp1ijp1])
            for d in pcon_data)
    end

    # Constraints B1(K,I): K=1..M, I=1..N-1
    # (1/(2DY))*Y(K,I,N-2) + (-2/DY)*Y(K,I,N-1) + (3/(2DY)+1)*Y(K,I,N) - U(K,I) = 0
    let c1y = T(c_1per2DY), cm2y = T(cm2perDY), c3y = T(c3per2DYp1)
        b1_data = [(yn2 = y_idx(K, I, N2), yn1 = y_idx(K, I, N1),
                    yn  = y_idx(K, I, N),  uk  = u_idx(K, I))
                   for K in 1:M for I in 1:N1]
        ExaModels.@add_con(c,
            c1y * x[d.yn2] + cm2y * x[d.yn1] + c3y * x[d.yn] - x[d.uk]
            for d in b1_data)
    end

    # Constraints B2(K,I): K=1..M, I=1..N-1
    # (1/(2DY))*Y(K,I,2) + (-2/DY)*Y(K,I,1) + (3/(2DY)+1)*Y(K,I,0) = 0
    let c1y = T(c_1per2DY), cm2y = T(cm2perDY), c3y = T(c3per2DYp1)
        b2_data = [(y2 = y_idx(K, I, 2), y1 = y_idx(K, I, 1), y0 = y_idx(K, I, 0))
                   for K in 1:M for I in 1:N1]
        ExaModels.@add_con(c,
            c1y * x[d.y2] + cm2y * x[d.y1] + c3y * x[d.y0]
            for d in b2_data)
    end

    # Constraints B3(K,J): K=1..M, J=1..N-1
    # (1/(2DX))*Y(K,2,J) + (-2/DX)*Y(K,1,J) + (3/(2DX)+1)*Y(K,0,J) = 0
    let c1x = T(c_1per2DX), cm2x = T(cm2perDX), c3x = T(c3per2DXp1)
        b3_data = [(y2j = y_idx(K, 2, J), y1j = y_idx(K, 1, J), y0j = y_idx(K, 0, J))
                   for K in 1:M for J in 1:N1]
        ExaModels.@add_con(c,
            c1x * x[d.y2j] + cm2x * x[d.y1j] + c3x * x[d.y0j]
            for d in b3_data)
    end

    # Constraints B4(K,J): K=1..M, J=1..N-1
    # (1/(2DX))*Y(K,N-2,J) + (-2/DX)*Y(K,N-1,J) + (3/(2DX)+1)*Y(K,N,J) = 0
    let c1x = T(c_1per2DX), cm2x = T(cm2perDX), c3x = T(c3per2DXp1)
        b4_data = [(yn2j = y_idx(K, N2, J), yn1j = y_idx(K, N1, J), ynj = y_idx(K, N, J))
                   for K in 1:M for J in 1:N1]
        ExaModels.@add_con(c,
            c1x * x[d.yn2j] + cm2x * x[d.yn1j] + c3x * x[d.ynj]
            for d in b4_data)
    end

    return ExaModels.ExaModel(c; kwargs...)
end
