# TRIMLOSS: Trim loss minimization in paper industry (MINLP treated as NLP)
# Source: Harjunkoski et al., Abo Akademi, Finland, 1996
# 142 variables: Y(J), BETA(J,L), Z(I,J,K), M(J), N(I,J)
# ELEMENT: SQRTXY(X,Y) = sqrt(X*Y)
function trimloss_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    # Constants
    NORD = T[12, 6, 15, 9]
    LL   = [15, 12, 9, 6]     # max multiple of cutting pattern
    KK   = [5, 5, 5, 4]       # max multiple per product
    B    = T[330, 365, 390, 435]  # width of orders
    BMAX = T(2000)
    DELTA = T(200)
    TEN  = T(10)

    # CC(J) = J/10
    CC = T[1/10, 2/10, 3/10, 4/10]
    # CCC(J,L) = L * C(J), C(J)=1.0, so CCC(J,L)=L
    # NLN17 rhs: NORD(I) + JJ = NORD(I) + 4
    NLN17_RHS = NORD .+ T(4)

    # Variable indexing:
    # Y(J), J=1..4:  indices 1..4
    # BETA(J,L), J=1..4, L=1..LL(J) (outer J, inner L): indices 5..46
    # Z(I,J,K), I=1..4, J=1..4, K=1..KK(I) (outer I, outer J, inner K): indices 47..122
    # M(J), J=1..4: indices 123..126
    # N(I,J), I=1..4, J=1..4: indices 127..142

    # Count BETA: sum LL = 15+12+9+6 = 42
    # Count Z: sum over I,J of KK(I) = 4 * sum(KK) = 4*19=76
    # Total: 4 + 42 + 76 + 4 + 16 = 142

    # Build bounds
    lvar = zeros(T, 142)
    uvar = fill(T(Inf), 142)

    # Y(J): 0 ≤ Y(J) ≤ 1
    for j in 1:4
        uvar[j] = T(1)
    end

    # BETA(J,L): 0 ≤ BETA ≤ 1
    beta_offset = 4
    for j in 1:4, _ in 1:LL[j]
        beta_offset += 1
        uvar[beta_offset] = T(1)
    end

    # Z(I,J,K): 0 ≤ Z ≤ 1
    z_offset = 4 + 42
    for _ in 1:4, _ in 1:4
        for i_prod in 1:4
            for _ in 1:KK[i_prod]
                z_offset += 1
                uvar[z_offset] = T(1)
            end
        end
    end
    # Hmm, order is I outer, J middle, K inner for Z
    # Rebuild: Z_offset starts at 46
    z_start = 46
    # Z(I,J,K): outer I=1..4, middle J=1..4, inner K=1..KK(I)
    # Index: z_start + (I-1)*4*max_KK is not uniform. Need cumulative.
    # z_idx(I,J,K) = z_start + sum_{i'<I} 4*KK[i'] + (J-1)*KK[I] + K
    function z_cumulative_base(i_val)
        base = z_start
        for ii in 1:i_val-1
            base += 4 * KK[ii]
        end
        return base
    end

    # Correct Z bounds (0 to 1)
    z_off = z_start
    for i in 1:4
        for _ in 1:4
            for _ in 1:KK[i]
                z_off += 1
                uvar[z_off] = T(1)
            end
        end
    end

    # M(J) and N(I,J): lvar=0, uvar=Inf (default)

    # START POINT: M(J)=1, N(I,J)=1 (from SIF XV TRIMLOSS M(J) 1.0, N(I,J) 1.0)
    x0 = zeros(T, 142)
    for j in 1:4
        x0[122 + j] = T(1)  # M(J)
    end
    for i in 1:4, j in 1:4
        x0[126 + 4 * (i - 1) + j] = T(1)  # N(I,J)
    end

    ExaModels.@add_var(c, x, 142; lvar = lvar, uvar = uvar, start = x0)

    # Objective: COST = sum_J CC(J)*Y(J) + sum_{J,L} L*BETA(J,L)
    # Y terms:
    ExaModels.@add_obj(c, CC[1] * x[1] + CC[2] * x[2] + CC[3] * x[3] + CC[4] * x[4])
    # BETA terms: sum_{J,L} L*BETA(J,L)
    beta_idx = 4
    for j in 1:4
        for l in 1:LL[j]
            beta_idx += 1
            ExaModels.@add_obj(c, T(l) * x[beta_idx])
        end
    end

    # Helper function for variable indices
    function y_idx(j)
        return j
    end
    function beta_idx_f(j, l)
        idx = 4
        for jj in 1:j-1
            idx += LL[jj]
        end
        idx += l
        return idx
    end
    function z_idx_f(i, j, k)
        idx = 46
        for ii in 1:i-1
            idx += 4 * KK[ii]
        end
        idx += (j - 1) * KK[i] + k
        return idx
    end
    function m_idx(j)
        return 122 + j
    end
    function n_idx(i, j)
        return 126 + 4 * (i - 1) + j
    end

    # Constraint (2): LIN2: sum_{J,L} (L-1)*BETA(J,L) <= -9
    # Wait: ZL LIN2 BETA(J,L) (L-1-L) = -L... let me re-read
    # "AI TEMP L; AM TEMP TEMP -1.0; ZL LIN2 BETA(J,L) TEMP"
    # TEMP = L; TEMP = -L. So coefficient is -L. LIN2 (L type): sum_{J,L} (-L)*BETA(J,L) <= rhs
    # CONSTANTS: LIN2 = -9.0. So: sum_{J,L} (-L)*BETA(J,L) - (-9) = sum(-L*BETA) + 9 as residual
    # For L type: group_value <= 0 means: sum(-L*BETA) + 9 <= 0 → sum(L*BETA) >= 9
    # Or equivalently: group value = sum(-L*BETA) <= rhs=-9 → sum(L*BETA) >= 9
    # CUTEst convention: con = group_value - rhs = sum(-L*BETA) - (-9) = sum(-L*BETA)+9
    # For L<=0: lcon=-Inf, ucon=0
    llin2 = T(0)
    for j in 1:4, l in 1:LL[j]
        llin2 += T(0)  # placeholder - we'll add dynamically
    end
    # Add as one expression
    lin2_expr = sum(T(-l) * x[beta_idx_f(j, l)] for j in 1:4 for l in 1:LL[j])
    # residual = lin2_expr - (-9) = lin2_expr + 9
    ExaModels.@add_con(c, sum(T(-l) * x[beta_idx_f(j, l)] for j in 1:4 for l in 1:LL[j]) + 9;
                       lcon = T(-Inf), ucon = T(0))

    # Constraint (3): LIN3(J): sum_{I,K} K*Z(I,J,K) + sum_I KK(I)*(-1)*Y(J) ... wait
    # "ZL LIN3(J) Z(I,J,K) K" then "AM TEMP KK(I) -1; ZL LIN3(J) Y(J) TEMP"
    # So LIN3(J): sum_{I,K} K*Z(I,J,K) - sum_I KK(I)*Y(J) <= 0
    for j in 1:4
        ExaModels.@add_con(c,
            sum(T(k) * x[z_idx_f(i, j, k)] for i in 1:4 for k in 1:KK[i]) -
            T(sum(KK)) * x[y_idx(j)];
            lcon = T(-Inf), ucon = T(0))
    end

    # Constraint (4): LIN4(J): sum_{I,K} B(I)*K*Z(I,J,K) - BMAX*Y(J) <= 0
    for j in 1:4
        ExaModels.@add_con(c,
            sum(B[i] * T(k) * x[z_idx_f(i, j, k)] for i in 1:4 for k in 1:KK[i]) -
            BMAX * x[y_idx(j)];
            lcon = T(-Inf), ucon = T(0))
    end

    # Constraint (5): LIN5(J): sum_{I,K} (-B(I)*K)*Z(I,J,K) + (BMAX-DELTA)*Y(J) <= 0
    for j in 1:4
        ExaModels.@add_con(c,
            sum(-B[i] * T(k) * x[z_idx_f(i, j, k)] for i in 1:4 for k in 1:KK[i]) +
            (BMAX - DELTA) * x[y_idx(j)];
            lcon = T(-Inf), ucon = T(0))
    end

    # Constraint (6): LIN6(J): sum_L (-L)*BETA(J,L) + Y(J) <= 0
    for j in 1:4
        ExaModels.@add_con(c,
            sum(T(-l) * x[beta_idx_f(j, l)] for l in 1:LL[j]) + x[y_idx(j)];
            lcon = T(-Inf), ucon = T(0))
    end

    # Constraint (7): LIN7(J): sum_L L*BETA(J,L) - LL(J)*Y(J) <= 0
    for j in 1:4
        ExaModels.@add_con(c,
            sum(T(l) * x[beta_idx_f(j, l)] for l in 1:LL[j]) - T(LL[j]) * x[y_idx(j)];
            lcon = T(-Inf), ucon = T(0))
    end

    # Constraint (8): LIN8(J): sum_L (2L^2-L)*BETA(J,L) + M(J) = 1 (rhs=1 from CONSTANTS)
    # "AI LR L; AA TEMP LR 2.0; A* TEMP TEMP LR; AM TEMP TEMP -1.0; ZE LIN8(J) BETA(J,L) TEMP"
    # TEMP = L; TEMP = 2.0 (as real); TEMP = 2*L; TEMP = 2*L*L; TEMP = -(2L^2)... wait
    # Actually: AI LR L (integer→real); AA TEMP LR 2.0 means TEMP = LR + 2.0 = L+2;
    # No wait - AA is "add arithmetic": TEMP = LR + 2.0 = L+2;
    # Then A* TEMP TEMP LR: TEMP = (L+2)*L = L^2+2L;
    # Then AM TEMP TEMP -1.0: TEMP = -(L^2+2L)
    # Hmm, let me re-read: "AA TEMP LR 2.0" -- in SIF R/RA/AA ops:
    # RA means TEMP = LR + 2.0 (add); "AA" is same as RA? Let me check SIF spec.
    # SIF: RE = set to value; RA = add; RM = multiply; R/ = divide; R* = multiply two parameters
    # "AA" is likely "RA" format with arithmetic: TEMP = LR + 2.0
    # Wait the SIF shows: "A  TEMP  LR  2.0" with first char 'A' in A-line? No it's AA.
    # In BATCH SIF context:
    # "AE ALPHA(J) 250.0" means ALPHA(J) = 250.0
    # "AA TEMP LR 2.0" means TEMP = LR + 2.0
    # So for LIN8: TEMP = L (from AI LR L; convert to real AA);
    #   Actually: "AI LR L" → LR = real(L); "AA TEMP LR 2.0" → TEMP = LR + 2.0 = L+2
    #   "A* TEMP TEMP LR" → TEMP = (L+2)*L = L^2+2L; "AM TEMP TEMP -1.0" → TEMP = -(L^2+2L)
    # So coefficient = -(L^2+2L) = -L(L+2)
    # LIN8(J): sum_L [-(L^2+2L)]*BETA(J,L) + M(J) = 1 (rhs from CONSTANTS LIN8(J)=1.0)
    # CUTEst residual = group_value - rhs = [sum(-(L^2+2L)*BETA) + M(J)] - 1
    for j in 1:4
        ExaModels.@add_con(c,
            sum(T(-(l^2 + 2l)) * x[beta_idx_f(j, l)] for l in 1:LL[j]) + x[m_idx(j)] - 1)
    end

    # Constraint (9): L9(I,J): sum_K [-(K^2+2K)]*Z(I,J,K) + N(I,J) = 1
    # Same arithmetic: KR = K; TEMP = K+2; TEMP = K(K+2) = K^2+2K; TEMP = -(K^2+2K)
    for i in 1:4
        for j in 1:4
            ExaModels.@add_con(c,
                sum(T(-(k^2 + 2k)) * x[z_idx_f(i, j, k)] for k in 1:KK[i]) + x[n_idx(i, j)] - 1)
        end
    end

    # Constraint (12): L12(I,J): sum_K Z(I,J,K) <= 1 (rhs=1, from CONSTANTS L12(I,J)=1.0)
    for i in 1:4, j in 1:4
        ExaModels.@add_con(c,
            sum(x[z_idx_f(i, j, k)] for k in 1:KK[i]) - 1;
            lcon = T(-Inf), ucon = T(0))
    end

    # Constraint (13): LIN13(J): sum_L BETA(J,L) <= 1 (rhs=1)
    for j in 1:4
        ExaModels.@add_con(c,
            sum(x[beta_idx_f(j, l)] for l in 1:LL[j]) - 1;
            lcon = T(-Inf), ucon = T(0))
    end

    # Constraint (14): LIN14(J): sum_L BETA(J+1,L) - sum_L BETA(J,L) <= 0, J=1..3
    for j in 1:3
        ExaModels.@add_con(c,
            sum(x[beta_idx_f(j + 1, l)] for l in 1:LL[j + 1]) -
            sum(x[beta_idx_f(j, l)] for l in 1:LL[j]);
            lcon = T(-Inf), ucon = T(0))
    end

    # Constraint (15): LIN15(J): Y(J+1) - Y(J) <= 0, J=1..3
    for j in 1:3
        ExaModels.@add_con(c, x[y_idx(j + 1)] - x[y_idx(j)]; lcon = T(-Inf), ucon = T(0))
    end

    # Constraint (16): LIN16(J): sum_L L*BETA(J,L) <= LL(J) (rhs from CONSTANTS = LL(J))
    for j in 1:4
        ExaModels.@add_con(c,
            sum(T(l) * x[beta_idx_f(j, l)] for l in 1:LL[j]) - T(LL[j]);
            lcon = T(-Inf), ucon = T(0))
    end

    # Constraint (17): NLN17(I) nonlinear part + linear part <= rhs
    # Linear part: sum_{J,L} L*BETA(J,L) + sum_{J,K} K*Z(I,J,K)
    # Nonlinear: -SQ(I,J) = -sqrt(M(J)*N(I,J)) for each J (coeff -1)
    # rhs: -(NORD(I) + JJ) = -(NORD(I)+4)
    # Group: L type: group_value <= 0
    # group_value = linear + nonlinear - rhs
    # = [sum L*BETA + sum K*Z - sum sqrt(M(J)*N(I,J))] - (-(NORD(I)+4))
    # = sum L*BETA + sum K*Z - sum sqrt(...) + NORD(I) + 4
    # The constraint lcon=-Inf, ucon=0
    for i in 1:4
        ExaModels.@add_con(c,
            sum(T(l) * x[beta_idx_f(j, l)] for j in 1:4 for l in 1:LL[j]) +
            sum(T(k) * x[z_idx_f(i, j, k)] for j in 1:4 for k in 1:KK[i]) -
            sum(sqrt(x[m_idx(j)] * x[n_idx(i, j)]) for j in 1:4) +
            NLN17_RHS[i];
            lcon = T(-Inf), ucon = T(0))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
