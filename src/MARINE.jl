# MARINE: Marine species population dynamics fitting via ODE collocation.
# NE=8 species, NM=21 observations, NC=2 collocation points, NH=200 subintervals.
# Source: COPS 2 problem 6 (Dolan and More', 2000). SIF: N. Gould, Nov 2000.
# Classification OOR2-AN-V-V
#
# Variables (0-indexed conceptually, 1-indexed in Julia):
#   M(s) for s=1..NE         → m_idx(s)
#   G(s) for s=1..NE-1       → g_idx(s)
#   V(i,s) for i=1..NH, s=1..NE  → v_idx(i,s)
#   W(i,k,s) for i=1..NH, k=1..NC, s=1..NE  → w_idx(i,k,s)
#   U(i,j,s) for i=1..NH, j=1..NC, s=1..NE  → u_idx(i,j,s)
#   DU(i,j,s) for i=1..NH, j=1..NC, s=1..NE → du_idx(i,j,s)
function marine_model(NH = 200; T = Float64, backend = nothing, kwargs...)
    NE = 8   # number of equations / species
    NM = 21  # number of observations
    NC = 2   # number of collocation points

    # Legendre polynomial roots (NC=2)
    RHO = T[0.7886751346, 0.2113248654]

    # Factorials: fact(k) for k = 0..NC
    fact = T[1.0, 1.0, 2.0]  # fact[1]=fact(0)=1, fact[2]=fact(1)=1, fact[3]=fact(2)=2

    # H = TF / NH, TF = tau(NM) = 10.0
    TF = T(10.0)
    H  = TF / T(NH)

    # Observation times tau(j) for j=1..NM
    tau = T[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
            5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

    # itau(j) = min(NH, floor(tau(j)/H) + 1)
    itau = [min(NH, floor(Int, tau[j] / H) + 1) for j in 1:NM]

    # Stage abundance measurements Z(j,s) for j=1..NM, s=1..NE
    Zmat = T[
        20000.0  17000.0  10000.0  15000.0  12000.0   9000.0   7000.0   3000.0;
        12445.0  15411.0  13040.0  13338.0  13484.0   8426.0   6615.0   4022.0;
         7705.0  13074.0  14623.0  11976.0  12453.0   9272.0   6891.0   5020.0;
         4664.0   8579.0  12434.0  12603.0  11738.0   9710.0   6821.0   5722.0;
         2977.0   7053.0  11219.0  11340.0  13665.0   8534.0   6242.0   5695.0;
         1769.0   5054.0  10065.0  11232.0  12112.0   9600.0   6647.0   7034.0;
          943.0   3907.0   9473.0  10334.0  11115.0   8826.0   6842.0   7348.0;
          581.0   2624.0   7421.0  10297.0  12427.0   8747.0   7199.0   7684.0;
          355.0   1744.0   5369.0   7748.0  10057.0   8698.0   6542.0   7410.0;
          223.0   1272.0   4713.0   6869.0   9564.0   8766.0   6810.0   6961.0;
          137.0    821.0   3451.0   6050.0   8671.0   8291.0   6827.0   7525.0;
           87.0    577.0   2649.0   5454.0   8430.0   7411.0   6423.0   8388.0;
           49.0    337.0   2058.0   4115.0   7435.0   7627.0   6268.0   7189.0;
           32.0    228.0   1440.0   3790.0   6474.0   6658.0   5859.0   7467.0;
           17.0    168.0   1178.0   3087.0   6524.0   5880.0   5562.0   7144.0;
           11.0     99.0    919.0   2596.0   5360.0   5762.0   4480.0   7256.0;
            7.0     65.0    647.0   1873.0   4556.0   5058.0   4944.0   7538.0;
            4.0     44.0    509.0   1571.0   4009.0   4527.0   4233.0   6649.0;
            2.0     27.0    345.0   1227.0   3677.0   4229.0   3805.0   6378.0;
            1.0     20.0    231.0    934.0   3197.0   3695.0   3159.0   6454.0;
            1.0     12.0    198.0    707.0   2562.0   3163.0   3232.0   5566.0;
    ]  # Zmat[j, s]

    # ── Variable index functions ───────────────────────────────────────────
    # M(s): s=1..NE, 1..NE
    m_idx(s) = s
    # G(s): s=1..NE-1
    g_idx(s) = NE + s
    base_V  = NE + (NE - 1)                    # after M, G
    # V(i,s): i=1..NH, s=1..NE → base_V + (i-1)*NE + s
    v_idx(i, s) = base_V + (i - 1) * NE + s
    base_W  = base_V + NH * NE                  # after V
    # W(i,k,s): i=1..NH, k=1..NC, s=1..NE → base_W + (i-1)*NC*NE + (k-1)*NE + s
    w_idx(i, k, s) = base_W + (i - 1) * NC * NE + (k - 1) * NE + s
    base_U  = base_W + NH * NC * NE             # after W
    # U(i,j,s): i=1..NH, j=1..NC, s=1..NE → base_U + (i-1)*NC*NE + (j-1)*NE + s
    u_idx(i, j, s) = base_U + (i - 1) * NC * NE + (j - 1) * NE + s
    base_DU = base_U + NH * NC * NE             # after U
    # DU(i,j,s): same layout
    du_idx(i, j, s) = base_DU + (i - 1) * NC * NE + (j - 1) * NE + s

    NV = base_DU + NH * NC * NE   # total variables

    # ── Bounds ────────────────────────────────────────────────────────────
    # Default: lvar=0, uvar=Inf (XR DEFAULT in SIF means all free from other bounds)
    # Actually, "XR MARINE 'DEFAULT'" means all variables are free (FR)
    # Then M(s) and G(s) have XL (>=0)
    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    for s in 1:NE
        lvar[m_idx(s)] = T(0)
    end
    for s in 1:NE-1
        lvar[g_idx(s)] = T(0)
    end

    # ── Start point ──────────────────────────────────────────────────────
    # SIF: XV MARINE 'DEFAULT' 1.0 sets all variables to 1.0.
    # Then specific XV entries override V(i,s) and U(i,j,s) for s=1..NE.
    # The SIF START POINT sets:
    #   - For i=1..itau[1] and all s=1..NE: V(i,s)=Z(1,s), U(i,j,s)=Z(1,s)
    #   - For K=2..NM-1, i=prev_end+1..itau[K], s=1..3: V(i,s)=Z(K,s), U(i,j,s)=Z(K,s)
    #   - For s=4..8 in intermediate K: kept at DEFAULT=1.0
    #   - For i=itau[NM-1]+1..NH and all s: V(i,s)=Z(NM,s), U(i,j,s)=Z(NM,s)
    # For simplicity, we initialize all to 1.0 then set known values.
    x0 = ones(T, NV)
    # First segment: all species
    for i in 1:itau[1]
        for s in 1:NE
            x0[v_idx(i, s)] = T(Zmat[1, s])
            for jj in 1:NC
                x0[u_idx(i, jj, s)] = T(Zmat[1, s])
            end
        end
    end
    # Middle segments: only s=1..3 are set in SIF
    for k_obs in 2:NM-1
        i_from = itau[k_obs-1] + 1
        i_to   = itau[k_obs]
        for i in i_from:i_to
            for s in 1:3  # only s=1,2,3 set in SIF
                x0[v_idx(i, s)] = T(Zmat[k_obs, s])
                for jj in 1:NC
                    x0[u_idx(i, jj, s)] = T(Zmat[k_obs, s])
                end
            end
        end
    end
    # Last segment: all species
    for i in itau[NM-1]+1:NH
        for s in 1:NE
            x0[v_idx(i, s)] = T(Zmat[NM, s])
            for jj in 1:NC
                x0[u_idx(i, jj, s)] = T(Zmat[NM, s])
            end
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # ── Objective: sum_{j=1}^{NM} sum_{s=1}^{NE} OBJ(j,s)^2 ─────────────
    # L2 group type: contribution = v_g^2
    # OBJ(j,s) = V(itau(j),s) + sum_{k=1}^{NC} W(itau(j),k,s)*COEF_k(j) - Z(j,s)
    # where COEF_k(j) = (tau(j) - t(itau(j)))^k / (fact(k) * h^(k-1))
    #       t(i) = (i-1)*H
    obj_data = [
        (
            vi   = v_idx(itau[j], s),
            w1   = w_idx(itau[j], 1, s),
            w2   = w_idx(itau[j], 2, s),
            c1   = let diff = tau[j] - (itau[j] - 1) * H
                       diff / (fact[2] * H^0)   # k=1: fact(1)*h^0 = 1
                   end,
            c2   = let diff = tau[j] - (itau[j] - 1) * H
                       diff^2 / (fact[3] * H^1)  # k=2: fact(2)*h^1 = 2H
                   end,
            zjs  = T(Zmat[j, s]),
        )
        for j in 1:NM for s in 1:NE
    ]
    ExaModels.@add_obj(c,
        (x[d.vi] + d.c1 * x[d.w1] + d.c2 * x[d.w2] - d.zjs)^2
        for d in obj_data)

    # ── U constraints: U(i,j,s) - V(i,s) - h*sum_k W(i,k,s)*rho(j)^k/fact(k) = 0
    # NC=2: coef_k_j = H * rho(j)^k / fact(k)
    # k=1: H * rho(j)^1 / fact(1) = H * rho(j)
    # k=2: H * rho(j)^2 / fact(2) = H * rho(j)^2 / 2
    u_con_data = [
        (
            u  = u_idx(i, j, s),
            vi = v_idx(i, s),
            w1 = w_idx(i, 1, s),
            w2 = w_idx(i, 2, s),
            c1 = H * RHO[j],
            c2 = H * RHO[j]^2 / T(2),
        )
        for i in 1:NH for j in 1:NC for s in 1:NE
    ]
    ExaModels.@add_con(c,
        x[d.u] - x[d.vi] - d.c1 * x[d.w1] - d.c2 * x[d.w2]
        for d in u_con_data)

    # ── DU constraints: DU(i,j,s) - sum_k W(i,k,s)*rho(j)^(k-1)/fact(k-1) = 0
    # NC=2:
    # k=1: rho(j)^0 / fact(0) = 1
    # k=2: rho(j)^1 / fact(1) = rho(j)
    du_con_data = [
        (
            du = du_idx(i, j, s),
            w1 = w_idx(i, 1, s),
            w2 = w_idx(i, 2, s),
            c2 = RHO[j],          # coefficient for k=2
        )
        for i in 1:NH for j in 1:NC for s in 1:NE
    ]
    ExaModels.@add_con(c,
        x[d.du] - x[d.w1] - d.c2 * x[d.w2]
        for d in du_con_data)

    # ── Continuity: V(i,s) - V(i+1,s) + h*sum_{k=1}^{NC} W(i,k,s)/fact(k) = 0
    # NC=2: h/fact(1) = h, h/fact(2) = h/2
    cont_data = [
        (
            vi  = v_idx(i, s),
            vi1 = v_idx(i + 1, s),
            w1  = w_idx(i, 1, s),
            w2  = w_idx(i, 2, s),
        )
        for i in 1:NH-1 for s in 1:NE
    ]
    ExaModels.@add_con(c,
        x[d.vi] - x[d.vi1] + H * x[d.w1] + (H / T(2)) * x[d.w2]
        for d in cont_data)

    # ── Collocation CO(i,j,s): DU + nonlinear terms = 0 ─────────────────
    # CO(i,j,s) linear part: DU(i,j,s). Nonlinear: PROD elements.
    #
    # s=1: DU(i,j,1) + G(1)*U(i,j,1) + M(1)*U(i,j,1) = 0
    # s=2..NE-1: DU(i,j,s) - G(s-1)*U(i,j,s-1) + G(s)*U(i,j,s) + M(s)*U(i,j,s) = 0
    # s=NE: DU(i,j,NE) - G(NE-1)*U(i,j,NE-1) + M(NE)*U(i,j,NE) = 0
    #
    # In ExaModels: bilinear products x[G_s]*x[U_ijs] must be written directly

    # s=1
    co_s1_data = [(du=du_idx(i,j,1), g1=g_idx(1), u1=u_idx(i,j,1), m1=m_idx(1))
                  for i in 1:NH for j in 1:NC]
    ExaModels.@add_con(c,
        x[d.du] + x[d.g1]*x[d.u1] + x[d.m1]*x[d.u1]
        for d in co_s1_data)

    # s=2..NE-1
    co_smid_data = [(du=du_idx(i,j,s), gs1=g_idx(s-1), gs=g_idx(s), ms=m_idx(s),
                     us1=u_idx(i,j,s-1), us=u_idx(i,j,s))
                    for i in 1:NH for j in 1:NC for s in 2:NE-1]
    ExaModels.@add_con(c,
        x[d.du] - x[d.gs1]*x[d.us1] + x[d.gs]*x[d.us] + x[d.ms]*x[d.us]
        for d in co_smid_data)

    # s=NE
    co_sNE_data = [(du=du_idx(i,j,NE), gs1=g_idx(NE-1), mNE=m_idx(NE),
                    us1=u_idx(i,j,NE-1), uNE=u_idx(i,j,NE))
                   for i in 1:NH for j in 1:NC]
    ExaModels.@add_con(c,
        x[d.du] - x[d.gs1]*x[d.us1] + x[d.mNE]*x[d.uNE]
        for d in co_sNE_data)

    return ExaModels.ExaModel(c; kwargs...)
end
