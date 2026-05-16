# FLOSP2HH/HL/HM/TH/TL/TM: 2D base flow in an inclined enclosure
# Three unknowns (vorticity OM, temperature PH, stream function PS) per grid node.
# Grid: I,J in -M..M (2M+1)^2 nodes, default M=15.
#
# The 6 variants differ only in Rayleigh number (RA) and boundary type (H=heat flux, T=temperature):
#   HH: RA=1e7, heat flux;  HM: RA=1e5, heat flux;  HL: RA=1e3, heat flux
#   TH: RA=1e7, temperature;TM: RA=1e5, temperature; TL: RA=1e3, temperature
#
# Source: J.N. Shadid, Ph.D. Thesis, U. Minnesota, 1989.
# SIF input: Nick Gould, August 1993.

function _flosp2_model(M::Int, RA::Float64, A1::Float64, A2::Float64, A3::Float64,
                       B1::Float64, B2::Float64, B3::Float64,
                       F1::Float64, F2::Float64, F3::Float64,
                       G1::Float64, G2::Float64, G3::Float64;
                       T = Float64, backend = nothing, kwargs...)
    # AX = 1.0, THETA = pi/2 => SINTHETA=1, COSTHETA≈0
    AX = 1.0; AXX = AX^2  # = 1.0
    pi_val = 4.0 * atan(1.0)
    THETA  = pi_val * 0.5
    COSTH  = cos(THETA)   # ≈ 0
    SINTH  = sin(THETA)   # = 1

    inv_H  = Float64(M)           # 1/H = M (H = 1/M)
    H      = 1.0 / inv_H
    inv_H2 = inv_H^2              # M^2

    PI1 = -0.5 * AX  * RA * COSTH   # ≈ 0
    PI2 =  0.5 * AXX * RA * SINTH   # = 0.5*RA

    # Derived coefficients
    coef_OM_lap  = T(-2.0 * inv_H2 - 2.0 * AXX * inv_H2)
    coef_OM_hi   = T(inv_H2)
    coef_OM_hj   = T(AXX * inv_H2)
    coef_PH_xi   = T(-PI1 / (2.0 * H))                      # ≈ 0
    coef_PH_xj   = T(-PI2 / (2.0 * H))
    coef_PS_lap  = T(-2.0 * inv_H2 - 2.0 * AXX * inv_H2)
    coef_PS_hi   = T(inv_H2)
    coef_PS_hj   = T(AXX * inv_H2)
    coef_OM_self = T(AXX / 4.0)
    coef_AX4H2   = T(AX * inv_H2 / 4.0)

    # Boundary condition coefficients
    c2A1H  = T(2.0 * A1 * inv_H);  cA2    = T(A2)
    c2B1H  = T(2.0 * B1 * inv_H);  cB2    = T(B2)
    c2F1AH = T(2.0 * F1 / AX * inv_H); cF2 = T(F2)
    c2G1AH = T(2.0 * G1 / AX * inv_H); cG2 = T(G2)
    cA3 = T(A3); cB3 = T(B3); cF3 = T(F3); cG3 = T(G3)
    c2H  = T(2.0 * inv_H)
    c2AH = T(2.0 / AX * inv_H)

    # Variable ordering: DO J=-M..M, DO I=-M..M => (OM(I,J), PH(I,J), PS(I,J))
    # Julia 1-indexed: node (I,J) -> flat index (J+M)*(2M+1) + (I+M) + 1
    # Variables: 1-based, for each node: OM=3*k-2, PH=3*k-1, PS=3*k
    NM1 = 2*M+1
    function node_base(i, j)  # i,j in -M..M
        3 * ((j + M) * NM1 + (i + M)) + 1  # first var (OM) at this node
    end
    om_idx(i, j) = node_base(i, j)
    ph_idx(i, j) = node_base(i, j) + 1
    ps_idx(i, j) = node_base(i, j) + 2

    NV = 3 * NM1^2

    # Bounds: all free ('DEFAULT' FR), except PS on boundary = 1 (XX ... 1.0)
    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    x0   = zeros(T, NV)

    # Fix PS on boundary: K=-M..M, PS(K,-M)=PS(-M,K)=PS(K,M)=PS(M,K)=1.0
    for K in -M:M
        for idx in [ps_idx(K, -M), ps_idx(-M, K), ps_idx(K, M), ps_idx(M, K)]
            lvar[idx] = T(1); uvar[idx] = T(1)
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # === Interior constraints (I,J in -M+1..M-1) ===
    # Node ordering: J outer (-M+1..M-1), I inner (-M+1..M-1) — matches SIF DO loop order.

    int_data = [(
        om  = om_idx(i, j),
        omp = om_idx(i+1, j),  omm = om_idx(i-1, j),
        omjp = om_idx(i, j+1), omjm = om_idx(i, j-1),
        ph  = ph_idx(i, j),
        php = ph_idx(i+1, j),  phm = ph_idx(i-1, j),
        phjp = ph_idx(i, j+1), phjm = ph_idx(i, j-1),
        ps  = ps_idx(i, j),
        psp = ps_idx(i+1, j),  psm = ps_idx(i-1, j),
        psjp = ps_idx(i, j+1), psjm = ps_idx(i, j-1),
    ) for j in -M+1:M-1 for i in -M+1:M-1]

    # S(I,J): stream function equation — linear
    ExaModels.@add_con(c,
        coef_OM_lap * x[d.om] + coef_OM_hi * (x[d.omp] + x[d.omm]) +
        coef_OM_hj * (x[d.omjp] + x[d.omjm]) +
        coef_PH_xi * (x[d.php] - x[d.phm]) +
        coef_PH_xj * (x[d.phjp] - x[d.phjm])
        for d in int_data)

    # V(I,J): vorticity equation — linear
    ExaModels.@add_con(c,
        coef_PS_lap * x[d.ps] + coef_PS_hi * (x[d.psp] + x[d.psm]) +
        coef_PS_hj * (x[d.psjp] + x[d.psjm]) +
        coef_OM_self * x[d.om]
        for d in int_data)

    # E(I,J): thermal energy equation — nonlinear
    ExaModels.@add_con(c,
        coef_PS_lap * x[d.ph] + coef_PS_hi * (x[d.php] + x[d.phm]) +
        coef_PS_hj * (x[d.phjp] + x[d.phjm]) +
        (-coef_AX4H2) * (x[d.psjp] - x[d.psjm]) * (x[d.php] - x[d.phm]) +
        coef_AX4H2  * (x[d.psp]  - x[d.psm])  * (x[d.phjp] - x[d.phjm])
        for d in int_data)

    # === Boundary constraints ===
    # Constraint creation order matches CUTEst's SIF group first-declaration order:
    #   K = -M (corners): T(-M,M), T(-M,-M), T(M,-M), V(-M,M), V(-M,-M), V(M,-M)   [6 new]
    #   K = -M+1..-1, 1..M-1 (non-corner): T(K,M), T(K,-M), T(M,K), T(-M,K),       [8 per K]
    #                                        V(K,M), V(K,-M), V(M,K), V(-M,K)
    #   K = M (corners): T(M,M), V(M,M)                                              [2 new]
    #
    # Corner RHS values (last Z-command in CONSTANTS section wins):
    #   T(-M,M):  A3 (from K=-M T(K,M)) then G3 (from K=M T(-M,K)) → RHS=G3
    #   T(-M,-M): B3 (from K=-M T(K,-M)) then G3 (from K=-M T(-M,K)) → RHS=G3
    #   T(M,-M):  F3 (from K=-M T(M,K)) then B3 (from K=M T(K,-M)) → RHS=B3
    #   T(M,M):   A3 (from K=M T(K,M)) then F3 (from K=M T(M,K)) → RHS=F3

    # Pre-compute corner variable indices
    ph_Mm_M   = ph_idx(-M, M);   ph_Mm1_M  = ph_idx(-M+1, M)   # T(-M,M) phG side
    ph_Mm_Mm1 = ph_idx(-M, M-1)                                  # T(-M,M) phA side
    ps_Mm_M   = ps_idx(-M, M);   ps_Mm_M1  = ps_idx(-M, M-1)    # V(-M,M) ps side

    ph_Mm_mM  = ph_idx(-M, -M);  ph_Mm1_mM = ph_idx(-M+1, -M)  # T(-M,-M) phG side
    ph_Mm_mM1 = ph_idx(-M, -M+1)                                 # T(-M,-M) phB side
    ps_Mm_mM  = ps_idx(-M, -M);  ps_Mm_mM1 = ps_idx(-M, -M+1)   # V(-M,-M) ps side

    ph_M_mM   = ph_idx(M, -M);   ph_M1_mM  = ph_idx(M-1, -M)   # T(M,-M) phF side
    ph_M_mM1  = ph_idx(M, -M+1)                                  # T(M,-M) phB side
    ps_M_mM   = ps_idx(M, -M);   ps_M1_mM  = ps_idx(M-1, -M)    # V(M,-M) ps side

    ph_M_M    = ph_idx(M, M);    ph_M1_M   = ph_idx(M-1, M)     # T(M,M) phF side
    ph_M_M1   = ph_idx(M, M-1)                                   # T(M,M) phA side
    ps_M_M    = ps_idx(M, M);    ps_M1_M   = ps_idx(M-1, M)      # V(M,M) ps side

    # Corner T and V constraints: K=-M first, then K=M last
    # T(-M,M): combined from T(K=-M,M) [A side] + T(-M,K=M) [G side], RHS=G3
    ExaModels.@add_con(c,
        c2A1H*(x[ph_Mm_M] - x[ph_Mm_Mm1]) + cA2*x[ph_Mm_M] +
        c2G1AH*(x[ph_Mm1_M] - x[ph_Mm_M]) + cG2*x[ph_Mm_M] - cG3)

    # T(-M,-M): combined from T(K=-M,-M) [B side] + T(-M,K=-M) [G side], RHS=G3
    ExaModels.@add_con(c,
        c2B1H*(x[ph_Mm_mM1] - x[ph_Mm_mM]) + cB2*x[ph_Mm_mM] +
        c2G1AH*(x[ph_Mm1_mM] - x[ph_Mm_mM]) + cG2*x[ph_Mm_mM] - cG3)

    # T(M,-M): combined from T(M,K=-M) [F side] + T(K=M,-M) [B side], RHS=B3
    ExaModels.@add_con(c,
        c2F1AH*(x[ph_M_mM] - x[ph_M1_mM]) + cF2*x[ph_M_mM] +
        c2B1H*(x[ph_M_mM1] - x[ph_M_mM]) + cB2*x[ph_M_mM] - cB3)

    # V(-M,M): boundary vorticity, top-left corner
    ExaModels.@add_con(c,
        -c2H*x[ps_Mm_M] + c2H*x[ps_Mm_M1])

    # V(-M,-M): boundary vorticity, bottom-left corner
    ExaModels.@add_con(c,
        c2H*x[ps_Mm_mM1] - c2H*x[ps_Mm_mM])

    # V(M,-M): boundary vorticity, bottom-right corner
    ExaModels.@add_con(c,
        c2AH*x[ps_M1_mM] - c2AH*x[ps_M_mM])

    # Non-corner boundary: K = -M+1 to M-1
    # Each K contributes 8 constraints in order:
    # T(K,M), T(K,-M), T(M,K), T(-M,K), V(K,M), V(K,-M), V(M,K), V(-M,K)
    mid_bnd = [(k = k,
                ph_km  = ph_idx(k, M),   ph_km1 = ph_idx(k, M-1),    # T(K,M)
                ph_kmm = ph_idx(k, -M),  ph_kmm1 = ph_idx(k, -M+1),  # T(K,-M)
                ph_mk  = ph_idx(M, k),   ph_m1k = ph_idx(M-1, k),    # T(M,K)
                ph_mmk = ph_idx(-M, k),  ph_mm1k = ph_idx(-M+1, k),  # T(-M,K)
                ps_km  = ps_idx(k, M),   ps_km1 = ps_idx(k, M-1),    # V(K,M)
                ps_kmm = ps_idx(k, -M),  ps_kmm1 = ps_idx(k, -M+1),  # V(K,-M)
                ps_mk  = ps_idx(M, k),   ps_m1k = ps_idx(M-1, k),    # V(M,K)
                ps_mmk = ps_idx(-M, k),  ps_mm1k = ps_idx(-M+1, k),  # V(-M,K)
               ) for k in -M+1:M-1]

    # T(K,M): top boundary temperature
    ExaModels.@add_con(c,
        c2A1H*(x[d.ph_km] - x[d.ph_km1]) + cA2*x[d.ph_km] - cA3
        for d in mid_bnd)

    # T(K,-M): bottom boundary temperature
    ExaModels.@add_con(c,
        c2B1H*(x[d.ph_kmm1] - x[d.ph_kmm]) + cB2*x[d.ph_kmm] - cB3
        for d in mid_bnd)

    # T(M,K): right boundary temperature
    ExaModels.@add_con(c,
        c2F1AH*(x[d.ph_mk] - x[d.ph_m1k]) + cF2*x[d.ph_mk] - cF3
        for d in mid_bnd)

    # T(-M,K): left boundary temperature
    ExaModels.@add_con(c,
        c2G1AH*(x[d.ph_mm1k] - x[d.ph_mmk]) + cG2*x[d.ph_mmk] - cG3
        for d in mid_bnd)

    # V(K,M): top boundary vorticity
    ExaModels.@add_con(c,
        -c2H*x[d.ps_km] + c2H*x[d.ps_km1]
        for d in mid_bnd)

    # V(K,-M): bottom boundary vorticity
    ExaModels.@add_con(c,
        c2H*x[d.ps_kmm1] - c2H*x[d.ps_kmm]
        for d in mid_bnd)

    # V(M,K): right boundary vorticity
    ExaModels.@add_con(c,
        -c2AH*x[d.ps_mk] + c2AH*x[d.ps_m1k]
        for d in mid_bnd)

    # V(-M,K): left boundary vorticity
    ExaModels.@add_con(c,
        c2AH*x[d.ps_mm1k] - c2AH*x[d.ps_mmk]
        for d in mid_bnd)

    # Corner constraints for K=M: T(M,M), V(M,M)
    # T(M,M): combined from T(K=M,M) [A side] + T(M,K=M) [F side], RHS=F3
    ExaModels.@add_con(c,
        c2A1H*(x[ph_M_M] - x[ph_M_M1]) + cA2*x[ph_M_M] +
        c2F1AH*(x[ph_M_M] - x[ph_M1_M]) + cF2*x[ph_M_M] - cF3)

    # V(M,M): boundary vorticity, top-right corner
    ExaModels.@add_con(c,
        -c2H*x[ps_M_M] + c2H*x[ps_M1_M])

    return ExaModels.ExaModel(c; kwargs...)
end

# Heat flux boundary (A1=B1=F1=G1=1, A2=B2=F2=G2=0)
function flosp2hh_model(M = 15; T = Float64, backend = nothing, kwargs...)
    _flosp2_model(M, 1e7, 1.0,0.0,-1.0, 1.0,0.0,-1.0, 1.0,0.0,0.0, 1.0,0.0,0.0; T=T, backend=backend, kwargs...)
end

function flosp2hm_model(M = 15; T = Float64, backend = nothing, kwargs...)
    _flosp2_model(M, 1e5, 1.0,0.0,-1.0, 1.0,0.0,-1.0, 1.0,0.0,0.0, 1.0,0.0,0.0; T=T, backend=backend, kwargs...)
end

function flosp2hl_model(M = 15; T = Float64, backend = nothing, kwargs...)
    _flosp2_model(M, 1e3, 1.0,0.0,-1.0, 1.0,0.0,-1.0, 1.0,0.0,0.0, 1.0,0.0,0.0; T=T, backend=backend, kwargs...)
end

# Temperature boundary (A1=B1=F1=G1=0, A2=B2=F2=G2=1)
function flosp2th_model(M = 15; T = Float64, backend = nothing, kwargs...)
    _flosp2_model(M, 1e7, 0.0,1.0,0.0, 0.0,1.0,1.0, 0.0,1.0,0.0, 0.0,1.0,0.0; T=T, backend=backend, kwargs...)
end

function flosp2tm_model(M = 15; T = Float64, backend = nothing, kwargs...)
    _flosp2_model(M, 1e5, 0.0,1.0,0.0, 0.0,1.0,1.0, 0.0,1.0,0.0, 0.0,1.0,0.0; T=T, backend=backend, kwargs...)
end

function flosp2tl_model(M = 15; T = Float64, backend = nothing, kwargs...)
    _flosp2_model(M, 1e3, 0.0,1.0,0.0, 0.0,1.0,1.0, 0.0,1.0,0.0, 0.0,1.0,0.0; T=T, backend=backend, kwargs...)
end
