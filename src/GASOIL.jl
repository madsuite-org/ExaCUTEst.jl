# GASOIL: Catalytic cracking of gas oil (COPS problem 12)
# Collocation ODE problem with NH=400 intervals, NC=4 collocation points per interval
# NE=2 ODE equations, NM=21 measurements, NP=3 ODE parameters
# Variables: THETA(1..3), then for each I=1..NH: V(I,1..2), W(I,1..4,1..2),
#            U(I,1..4,1..2), DU(I,1..4,1..2)   [26 variables per interval]
# Source: E. Dolan and J. More', "Benchmarking Optimization Software with COPS", 2000
# SIF input: Nick Gould, November 2000
function gasoil_model(NH = 400; T = Float64, backend = nothing, kwargs...)
    NC = 4; NE = 2; NP = 3; NM = 21

    # Problem constants
    TF_f  = 0.95
    H_f   = TF_f / NH
    H     = T(H_f)

    # Legendre roots (NC=4)
    RHO_f = [0.0694318442, 0.3300094782, 0.6699905218, 0.9305681558]

    # Observation times
    TAU_f = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.150, 0.175, 0.20, 0.225,
             0.250, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95]

    # Measured concentrations Z(J,S) for J=1..NM, S=1..NE
    Z1_f = [1.0000, 0.8105, 0.6208, 0.5258, 0.4345, 0.3903, 0.3342, 0.3034,
            0.2735, 0.2405, 0.2283, 0.2071, 0.1669, 0.1530, 0.1339, 0.1265,
            0.1200, 0.0990, 0.0870, 0.0770, 0.0690]
    Z2_f = [0.0000, 0.2000, 0.2886, 0.3010, 0.3215, 0.3123, 0.2716, 0.2551,
            0.2258, 0.1959, 0.1789, 0.1457, 0.1198, 0.0909, 0.0719, 0.0561,
            0.0460, 0.0280, 0.0190, 0.0140, 0.0100]

    # Initial conditions BC(S)
    BC_f = [1.0, 0.0]

    # Factorials FACT(k) for k=0..NC (1-indexed: FACT[k+1] = k!)
    FACT_f = [1.0, 1.0, 2.0, 6.0, 24.0]  # FACT[1]=0!, FACT[2]=1!, ..., FACT[5]=4!

    # ITAU(J) = min(NH, floor(TAU(J)/H) + 1)
    ITAU = [min(NH, Int(floor(TAU_f[j] / H_f)) + 1) for j in 1:NM]

    # Variable indexing (Julia 1-based)
    # THETA(i): index i for i=1..NP=3
    # For interval I=1..NH, per_I=26 variables starting at NP+(I-1)*per_I+1:
    #   V(I,S):     offset + S                   S=1..NE
    #   W(I,K,S):   offset + NE + (K-1)*NE + S   K=1..NC, S=1..NE
    #   U(I,J,S):   offset + NE+NC*NE + (J-1)*NE + S   J=1..NC, S=1..NE
    #   DU(I,J,S):  offset + NE+2*NC*NE + (J-1)*NE + S
    per_I = NE + NC*NE + NC*NE + NC*NE  # = 26

    v_idx(I, S)     = NP + (I-1)*per_I + S
    w_idx(I, K, S)  = NP + (I-1)*per_I + NE + (K-1)*NE + S
    u_idx(I, J, S)  = NP + (I-1)*per_I + NE + NC*NE + (J-1)*NE + S
    du_idx(I, J, S) = NP + (I-1)*per_I + NE + 2*NC*NE + (J-1)*NE + S

    NV = NP + NH * per_I

    # Bounds and starting point
    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    x0   = zeros(T, NV)

    # THETA(I) >= 0, start at 0
    for i in 1:NP; lvar[i] = T(0); end

    # V(1,S) fixed to BC(S)
    for S in 1:NE
        idx = v_idx(1, S)
        lvar[idx] = T(BC_f[S]); uvar[idx] = T(BC_f[S]); x0[idx] = T(BC_f[S])
    end

    # Initialize remaining vars from measurement interpolation
    for I in 1:NH
        for S in 1:NE
            x0[v_idx(I, S)] = T(BC_f[S])
        end
        for K in 1:NC, S in 1:NE
            x0[w_idx(I, K, S)] = T(0)
        end
        for J in 1:NC, S in 1:NE
            x0[u_idx(I, J, S)]  = T(BC_f[S])
            x0[du_idx(I, J, S)] = T(0)
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # === Objective: sum_{J,S} (V(ITAU(J),S) + sum_k W(ITAU(J),k,S)*coef_k - Z(J,S))^2 ===
    # Build data tuples: (vi, wi1..wi4, wc1..wc4, z)
    # where wc_k = (tau_j - t[itau_j])^k / (FACT[k+1] * H^(k-1))
    obj_data = [(
        vi  = v_idx(ITAU[J], S),
        wi1 = w_idx(ITAU[J], 1, S),
        wi2 = w_idx(ITAU[J], 2, S),
        wi3 = w_idx(ITAU[J], 3, S),
        wi4 = w_idx(ITAU[J], 4, S),
        wc1 = T(let DIFF = TAU_f[J] - H_f*(ITAU[J]-1); DIFF^1 / (FACT_f[2] * H_f^0) end),
        wc2 = T(let DIFF = TAU_f[J] - H_f*(ITAU[J]-1); DIFF^2 / (FACT_f[3] * H_f^1) end),
        wc3 = T(let DIFF = TAU_f[J] - H_f*(ITAU[J]-1); DIFF^3 / (FACT_f[4] * H_f^2) end),
        wc4 = T(let DIFF = TAU_f[J] - H_f*(ITAU[J]-1); DIFF^4 / (FACT_f[5] * H_f^3) end),
        z   = T(S == 1 ? Z1_f[J] : Z2_f[J]),
    ) for J in 1:NM for S in 1:NE]

    ExaModels.@add_obj(c,
        (x[d.vi] + d.wc1*x[d.wi1] + d.wc2*x[d.wi2] +
         d.wc3*x[d.wi3] + d.wc4*x[d.wi4] - d.z)^2
        for d in obj_data)

    # === U(I,J,S) constraints: -U + V + H*sum_k W*rho^k/fact[k] = 0 ===
    u_con_data = [(
        u   = u_idx(I, J, S),
        v   = v_idx(I, S),
        wi1 = w_idx(I, 1, S),
        wi2 = w_idx(I, 2, S),
        wi3 = w_idx(I, 3, S),
        wi4 = w_idx(I, 4, S),
        wc1 = T(H_f * RHO_f[J]^1 / FACT_f[2]),
        wc2 = T(H_f * RHO_f[J]^2 / FACT_f[3]),
        wc3 = T(H_f * RHO_f[J]^3 / FACT_f[4]),
        wc4 = T(H_f * RHO_f[J]^4 / FACT_f[5]),
    ) for I in 1:NH for J in 1:NC for S in 1:NE]

    ExaModels.@add_con(c,
        -x[d.u] + x[d.v] +
        d.wc1*x[d.wi1] + d.wc2*x[d.wi2] +
        d.wc3*x[d.wi3] + d.wc4*x[d.wi4]
        for d in u_con_data)

    # === DU(I,J,S) constraints: -DU + sum_k W*rho^(k-1)/fact[k-1] = 0 ===
    du_con_data = [(
        du  = du_idx(I, J, S),
        wi1 = w_idx(I, 1, S),
        wi2 = w_idx(I, 2, S),
        wi3 = w_idx(I, 3, S),
        wi4 = w_idx(I, 4, S),
        wc1 = T(RHO_f[J]^0 / FACT_f[1]),
        wc2 = T(RHO_f[J]^1 / FACT_f[2]),
        wc3 = T(RHO_f[J]^2 / FACT_f[3]),
        wc4 = T(RHO_f[J]^3 / FACT_f[4]),
    ) for I in 1:NH for J in 1:NC for S in 1:NE]

    ExaModels.@add_con(c,
        -x[d.du] +
        d.wc1*x[d.wi1] + d.wc2*x[d.wi2] +
        d.wc3*x[d.wi3] + d.wc4*x[d.wi4]
        for d in du_con_data)

    # === Continuity C(I,S): V(I,S) - V(I+1,S) + sum_j W(I,j,S)*H/fact[j] = 0 ===
    cont_data = [(
        vi  = v_idx(I, S),
        vi1 = v_idx(I+1, S),
        wi1 = w_idx(I, 1, S),
        wi2 = w_idx(I, 2, S),
        wi3 = w_idx(I, 3, S),
        wi4 = w_idx(I, 4, S),
        wc1 = T(H_f / FACT_f[2]),
        wc2 = T(H_f / FACT_f[3]),
        wc3 = T(H_f / FACT_f[4]),
        wc4 = T(H_f / FACT_f[5]),
    ) for I in 1:NH-1 for S in 1:NE]

    ExaModels.@add_con(c,
        x[d.vi] - x[d.vi1] +
        d.wc1*x[d.wi1] + d.wc2*x[d.wi2] +
        d.wc3*x[d.wi3] + d.wc4*x[d.wi4]
        for d in cont_data)

    # === Collocation CO(I,J,1): DU(I,J,1) + (theta1+theta3)*U(I,J,1)^2 = 0 ===
    # === Collocation CO(I,J,2): DU(I,J,2) - theta1*U(I,J,1)^2 + theta2*U(I,J,2) = 0 ===
    col_data = [(
        du1 = du_idx(I, J, 1),
        du2 = du_idx(I, J, 2),
        u11 = u_idx(I, J, 1),
        u12 = u_idx(I, J, 2),
    ) for I in 1:NH for J in 1:NC]

    ExaModels.@add_con(c,
        x[d.du1] + (x[1] + x[3]) * x[d.u11]^2
        for d in col_data)

    ExaModels.@add_con(c,
        x[d.du2] - x[1] * x[d.u11]^2 + x[2] * x[d.u12]
        for d in col_data)

    return ExaModels.ExaModel(c; kwargs...)
end
