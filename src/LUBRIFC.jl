# LUBRIFC: Corrected elastodynamic lubrication problem by Kostreva
# QOR2-MN-V-V: NN=250, N=5*NN=1250, NV=3N+1=3751 variables
# Source: Kostreva (1984), corrected by Leyffer (2000). SIF: Ph. Toint, June 1990.
#
# Variable layout (CUTEst ordering from VARIABLES section):
#   x[1]         = K (free)
#   x[2..N+2]    = P(0),P(2),...,P(2N) with P(2i)→x[i+2] (P(0)=P(2N)=0 fixed; others in [0,3])
#   x[N+3..2N+2] = H(1),H(3),...,H(2N-1) with H(2j-1)→x[N+2+j] (free)
#   x[2N+3..3N+1]= R(2),R(4),...,R(2N-2) with R(2i)→x[2N+2+i] (lvar=0, no upper)
#
# Groups:
#   R(0): equality constraint sum_{i=1}^{N-1} P(2i)*(2DX/PI) + P(2N)*(DX/PI) = 1
#         P(2N)=0 so: sum_{i=1}^{N-1} P(2i)*(2DX/PI) = 1
#   COMPL: XN (objective) = sum_{i=1}^{N-1} P(2i)*R(2i)
#   DR(I): equality for I=2,4,...,2N-2:
#     H(I+1)*L/DX - H(I-1)*L/DX - R(I)
#     + (1/DX2)*F_REY(PA=P(I-2),H=H(I-1),PB=P(I))
#     - (1/DX2)*F_REY(PA=P(I),H=H(I+1),PB=P(I+2)) = 0
#     where F_REY(PA,H,PB) = (PA-PB)*H^3*exp(-0.5*ALPHA*(PA+PB))
#   DH(J): equality for J=1,3,...,2N-1 (j=1..N):
#     K - H(J) + sum_{i=1}^{N-1} C[j,i]*P(2i) = -(xj^2+1) where xj = XA+(2j-1)*DX/2

function lubrifc_model(; T = Float64, backend = nothing, kwargs...)
    NN = 250
    N = 5 * NN   # 1250
    _2N = 2 * N  # 2500

    ALPHA = T(1.838)
    LAMBDA = T(1.642)
    XA = T(-3.0)
    XF = T(2.0)
    PI_val = T(3.1415926535)

    LEN = XF - XA                        # 5.0
    DX = LEN / T(N)                      # 0.004
    L_over_DX = LAMBDA / DX
    one_over_DX2 = T(1.0) / DX^2
    two_DX_over_PI = T(2.0) * DX / PI_val
    half_over_PI = T(0.5) / PI_val
    one_over_PI = T(1.0) / PI_val
    DX_over_2 = DX / T(2.0)

    NV = 3 * N + 1

    K_idx = 1
    P_idx(i) = i + 2            # P(2i) for i=0..N → x[2..N+2]
    H_idx(j) = N + 2 + j       # H(2j-1) for j=1..N → x[N+3..2N+2]
    R_idx(i) = 2N + 2 + i      # R(2i) for i=1..N-1 → x[2N+3..3N+1]

    # Bounds
    lvar = zeros(T, NV)
    uvar = fill(T(Inf), NV)
    lvar[K_idx] = T(-Inf); uvar[K_idx] = T(Inf)
    # P(0) and P(2N) fixed to 0 (already 0 from lvar init)
    uvar[P_idx(0)] = T(0)  # P(0) fixed: lvar[P_idx(0)]=0, uvar=0
    for i in 1:N-1
        uvar[P_idx(i)] = T(3.0)          # 0≤P(2i)≤3
    end
    uvar[P_idx(N)] = T(0)  # P(2N) fixed: uvar=0
    for j in 1:N
        lvar[H_idx(j)] = T(-Inf); uvar[H_idx(j)] = T(Inf)
    end
    # R(I): lvar=0 (default), no upper bound → [0, Inf]

    # Starting point
    x0 = zeros(T, NV)
    twoNN = 2 * NN   # 500
    fourNN = 4 * NN  # 1000
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

    # Precompute DH RHS: rhs[j] = -(xj^2 + 1)
    dh_rhs = T[-(( XA + T(2j-1) * DX_over_2 )^2 + T(1.0)) for j in 1:N]

    # Precompute C_mat[j, i] for j=1..N, i=1..N-1
    # C_mat[j,i] = coefficient of P(2i) in DH(J=2j-1) constraint
    # Algorithm from SIF (integration by parts for log kernel):
    C_mat = zeros(T, N, N-1)
    for j in 1:N
        J = 2j - 1
        Cvec = zeros(T, _2N + 2)  # Cvec[I] for even I=2..2N

        # Entry for I = -J (i.e., the "-J" offset in the first step of the SIF loop)
        # RI-J = -J → I-JDX = (-J)*DX/2 → contribution to C(2)
        val = T(-J) * DX_over_2
        if val != T(0)
            t1 = val * log(abs(val))
            Cvec[2] += t1 * half_over_PI
        end

        # Entry for I = 2 (second step: IA I-J -J 2)
        val = T(-J + 2) * DX_over_2
        if val != T(0)
            t1 = val * log(abs(val))
            Cvec[4] += t1 * one_over_PI
        end

        # Middle entries for I=4,6,...,2N-2 step 2
        for k in 2:N-1
            I = 2k
            IminJ = I - J
            val = T(IminJ) * DX_over_2
            if val != T(0)
                t1 = val * log(abs(val))
                coeff = t1 * one_over_PI
                if I + 2 <= _2N
                    Cvec[I+2] += coeff
                end
                if I - 2 >= 2
                    Cvec[I-2] -= coeff
                end
            end
        end

        # Last entry for I = 2N - J → contribution to C(2N-2)
        val = T(_2N - J) * DX_over_2
        if val != T(0)
            t1 = val * log(abs(val))
            coeff = t1 * half_over_PI
            if _2N - 2 >= 2
                Cvec[_2N - 2] -= coeff
            end
        end

        for i in 1:N-1
            C_mat[j, i] = Cvec[2i]
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: COMPL = sum_{i=1}^{N-1} P(2i)*R(2i)
    compl_data = [(pi = Int32(P_idx(i)), ri = Int32(R_idx(i))) for i in 1:N-1]
    ExaModels.@add_obj(c, x[d.pi] * x[d.ri] for d in compl_data)

    # Constraint R(0)=0: sum_{i=1}^{N-1} P(2i)*(2DX/PI) = 1
    r0_data = [(pi = Int32(P_idx(i)),) for i in 1:N-1]
    ExaModels.@add_con(c,
        sum(T(two_DX_over_PI) * x[r0_data[i].pi] for i in 1:N-1) - T(1.0);
        lcon = T(0.0), ucon = T(0.0))

    # Constraints DR(I) for i=1..N-1 (I=2i):
    # Linear: L/DX*H(I+1) - L/DX*H(I-1) - R(I)
    # Nonlinear: (1/DX2)*(F_REY_at_(I-1)) - (1/DX2)*(F_REY_at_(I+1))
    # For J=I-1=2i-1 (odd), j=i: PA=P(2i-2), H=H(2i-1), PB=P(2i)
    # For J=I+1=2i+1 (odd), j=i+1: PA=P(2i), H=H(2i+1), PB=P(2i+2)
    # H_idx: H(2j-1)→H_idx(j); H(2i-1)→H_idx(i); H(2i+1)→H_idx(i+1)
    # P_idx: P(2i-2)→P_idx(i-1); P(2i)→P_idx(i); P(2i+2)→P_idx(i+1)
    dr_data = [(pa1 = Int32(P_idx(i-1)), h1 = Int32(H_idx(i)),   pb1 = Int32(P_idx(i)),
                pa2 = Int32(P_idx(i)),   h2 = Int32(H_idx(i+1)), pb2 = Int32(P_idx(i+1)),
                hm1 = Int32(H_idx(i)),   hp1 = Int32(H_idx(i+1)), ri = Int32(R_idx(i)))
               for i in 1:N-1]
    ExaModels.@add_con(c,
        T(L_over_DX) * x[d.hp1] - T(L_over_DX) * x[d.hm1] - x[d.ri]
        + T(one_over_DX2) * (x[d.pa1] - x[d.pb1]) * x[d.h1]^3 *
            exp(T(-0.5) * ALPHA * (x[d.pa1] + x[d.pb1]))
        - T(one_over_DX2) * (x[d.pa2] - x[d.pb2]) * x[d.h2]^3 *
            exp(T(-0.5) * ALPHA * (x[d.pa2] + x[d.pb2]))
        for d in dr_data)

    # Constraints DH(J) for j=1..N:
    # K - H(J) + sum_{i=1}^{N-1} C_mat[j,i]*P(2i) - dh_rhs[j] = 0
    # Use sum() inside @add_con. C_mat[j,:] values are compile-time constants.
    dh_data = [(j = Int32(j), h_idx = Int32(H_idx(j)), rhs_val = T(dh_rhs[j])) for j in 1:N]
    # Store C_mat rows as a vector of SVector-like named tuples:
    # Since we can't use sum(C_mat[d.j,i]*x[P_idx(i)]) with d.j as runtime Int32 index into C_mat,
    # we precompute the sum for each j at model-build time and make it a @add_par.
    # But C_mat[j,i] are constants, not variables, so we CAN do:
    # @add_con(c, x[K_idx] - x[d.h_idx] + sum(C_mat[d.j, i] * x[P_idx(i)] for i in 1:N-1) - d.rhs_val for d in dh_data)
    # This should work because C_mat[d.j, i] accesses the precomputed array by runtime j,
    # but in the ExaModels macro context d.j is a value from the data array (treated as Int32 parameter).
    # The sum() will unroll to N-1 terms at compile time with i as a static range, but d.j varies.

    # Alternative: Use @add_par for C_mat and index it:
    ExaModels.@add_par(c, C_par, C_mat)  # C_par is an N×(N-1) parameter matrix

    ExaModels.@add_con(c,
        x[K_idx] - x[d.h_idx] +
        sum(C_par[d.j, i] * x[P_idx(i)] for i in 1:N-1) - d.rhs_val
        for d in dh_data)

    return ExaModels.ExaModel(c; kwargs...)
end
