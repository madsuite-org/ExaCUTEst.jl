# LUBRIF: Elastodynamic lubrication problem (contains an error in Reynolds equation)
# QOR2-MN-V-V: NN=250, N=5*NN=1250, NV=3N+1=3751 variables
# Source: Kostreva (1984). SIF: Ph. Toint, June 1990.
# NB: This has an error in ELEMENT USES: H and P(I+) are swapped in REY element.
# See LUBRIFC for the corrected version.
#
# The error: In ELEMENT USES, for element ER(J):
#   ZV ER(J) H   = P(I+)    [wrong: should be H(J)]
#   ZV ER(J) PB  = H(J)     [wrong: should be P(I+)]
#
# So F_REY in LUBRIF uses: PA=P(I-), H=P(I+), PB=H(J)
# vs LUBRIFC: PA=P(I-), H=H(J), PB=P(I+)
#
# All other structure is identical to LUBRIFC.

function lubrif_model(; T = Float64, backend = nothing, kwargs...)
    NN = 250
    N = 5 * NN   # 1250
    _2N = 2 * N  # 2500

    ALPHA = T(1.838)
    LAMBDA = T(1.642)
    XA = T(-3.0)
    XF = T(2.0)
    PI_val = T(3.1415926535)

    LEN = XF - XA
    DX = LEN / T(N)
    L_over_DX = LAMBDA / DX
    one_over_DX2 = T(1.0) / DX^2
    two_DX_over_PI = T(2.0) * DX / PI_val
    half_over_PI = T(0.5) / PI_val
    one_over_PI = T(1.0) / PI_val
    DX_over_2 = DX / T(2.0)

    NV = 3 * N + 1

    K_idx = 1
    P_idx(i) = i + 2
    H_idx(j) = N + 2 + j
    R_idx(i) = 2N + 2 + i

    lvar = zeros(T, NV)
    uvar = fill(T(Inf), NV)
    lvar[K_idx] = T(-Inf); uvar[K_idx] = T(Inf)
    uvar[P_idx(0)] = T(0)
    for i in 1:N-1
        uvar[P_idx(i)] = T(3.0)
    end
    uvar[P_idx(N)] = T(0)
    for j in 1:N
        lvar[H_idx(j)] = T(-Inf); uvar[H_idx(j)] = T(Inf)
    end

    x0 = zeros(T, NV)
    twoNN = 2 * NN
    fourNN = 4 * NN
    for i in 1:twoNN
        xi = XA + T(i) * DX
        x0[P_idx(i)] = T(0.02) * xi + T(0.06)
    end
    for i in twoNN+1:fourNN
        xi = XA + T(i) * DX
        v2 = T(1.0) - xi^2
        x0[P_idx(i)] = v2 > T(0) ? sqrt(v2) : T(0.0)
    end
    for j in 1:N
        xj = XA + T(2j - 1) * DX_over_2
        x0[H_idx(j)] = xj^2
    end

    dh_rhs = T[-(( XA + T(2j-1) * DX_over_2 )^2 + T(1.0)) for j in 1:N]

    C_mat = zeros(T, N, N-1)
    for j in 1:N
        J = 2j - 1
        Cvec = zeros(T, _2N + 2)

        val = T(-J) * DX_over_2
        if val != T(0)
            Cvec[2] += val * log(abs(val)) * half_over_PI
        end
        val = T(-J + 2) * DX_over_2
        if val != T(0)
            Cvec[4] += val * log(abs(val)) * one_over_PI
        end

        for k in 2:N-1
            I = 2k
            val = T(I - J) * DX_over_2
            if val != T(0)
                coeff = val * log(abs(val)) * one_over_PI
                if I + 2 <= _2N; Cvec[I+2] += coeff; end
                if I - 2 >= 2;   Cvec[I-2] -= coeff; end
            end
        end

        val = T(_2N - J) * DX_over_2
        if val != T(0)
            coeff = val * log(abs(val)) * half_over_PI
            if _2N - 2 >= 2; Cvec[_2N-2] -= coeff; end
        end

        for i in 1:N-1
            C_mat[j, i] = Cvec[2i]
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    compl_data = [(pi = Int32(P_idx(i)), ri = Int32(R_idx(i))) for i in 1:N-1]
    ExaModels.@add_obj(c, x[d.pi] * x[d.ri] for d in compl_data)

    r0_data = [(pi = Int32(P_idx(i)),) for i in 1:N-1]
    ExaModels.@add_con(c,
        sum(T(two_DX_over_PI) * x[r0_data[i].pi] for i in 1:N-1) - T(1.0);
        lcon = T(0.0), ucon = T(0.0))

    # DR(I) constraints with LUBRIF error: H and PB swapped in REY element.
    # For J=I-1 (j=i): PA=P(I-2), H=P(I) [wrong], PB=H(I-1) [wrong]
    # For J=I+1 (j=i+1): PA=P(I), H=P(I+2) [wrong], PB=H(I+1) [wrong]
    # F_REY_wrong(PA, H_wrong=P_Iplus, PB_wrong=H_J) = (PA - PB_wrong)*H_wrong^3*exp(-0.5*A*(PA+PB_wrong))
    # = (P(I-2) - H(I-1)) * P(I)^3 * exp(-0.5*A*(P(I-2) + H(I-1)))
    dr_data = [(pa1 = Int32(P_idx(i-1)), h1_wrong = Int32(P_idx(i)),   pb1_wrong = Int32(H_idx(i)),
                pa2 = Int32(P_idx(i)),   h2_wrong = Int32(P_idx(i+1)), pb2_wrong = Int32(H_idx(i+1)),
                hm1 = Int32(H_idx(i)),   hp1 = Int32(H_idx(i+1)), ri = Int32(R_idx(i)))
               for i in 1:N-1]
    ExaModels.@add_con(c,
        T(L_over_DX) * x[d.hp1] - T(L_over_DX) * x[d.hm1] - x[d.ri]
        + T(one_over_DX2) * (x[d.pa1] - x[d.pb1_wrong]) * x[d.h1_wrong]^3 *
            exp(T(-0.5) * ALPHA * (x[d.pa1] + x[d.pb1_wrong]))
        - T(one_over_DX2) * (x[d.pa2] - x[d.pb2_wrong]) * x[d.h2_wrong]^3 *
            exp(T(-0.5) * ALPHA * (x[d.pa2] + x[d.pb2_wrong]))
        for d in dr_data)

    dh_data = [(j = Int32(j), h_idx = Int32(H_idx(j)), rhs_val = T(dh_rhs[j])) for j in 1:N]
    ExaModels.@add_par(c, C_par, C_mat)
    ExaModels.@add_con(c,
        x[K_idx] - x[d.h_idx] +
        sum(C_par[d.j, i] * x[P_idx(i)] for i in 1:N-1) - d.rhs_val
        for d in dh_data)

    return ExaModels.ExaModel(c; kwargs...)
end
