# BATCH: Optimal Design of Multiproduct Batch Plant (MINLP treated as NLP)
# Source: Kocis & Grossmann, Ind. Eng. Chem. Res. 27 (1988), pp. 1407-1421
# Variables (log-space): N(j)=log(parallel units), V(j)=log(volume),
#   B(i)=log(batch size), TL(i)=log(cycle time), Y(k,j)=binary→continuous
# M=6 stages, N=5 products, NU=4 max parallel units
#
# sifdecoder produces 48 variables (not 46) because ELEMENT USES has a DO I 1 M=6
# loop for EXPC(I) referencing TL(I) and B(I), but TL and B are only declared for
# I=1..N=5 in VARIABLES; sifdecoder creates B(6) and TL(6) as extra free variables.
#
# Variable ordering (from sifdecoder):
#   pos 1..5   : TL(5),TL(4),TL(3),TL(2),TL(1)   → TL(i)=x[6-i]
#   pos 6..10  : B(5), B(4), B(3), B(2), B(1)     → B(i)=x[11-i]
#   pos 11..16 : V(5),V(6),V(4),V(3),V(2),V(1)    → use VPOS[j]
#   pos 17..22 : N(6),N(5),N(4),N(3),N(2),N(1)    → N(j)=x[23-j]
#   pos 23..46 : Y(k,j) k=1..4, j=1..6 inner-k    → Y(k,j)=x[22+(j-1)*4+k]
#   pos 47..48 : B(6), TL(6)  (unused extra vars)
function batch_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    # Precomputed constants
    LOGNU = log(T(4))
    VL    = log(T(300))
    VU    = log(T(3000))
    H     = T(6000)

    # S(I,J) = log(size factors)  [I=product 1..5, J=stage 1..6]
    S = T[log(7.9)  log(2.0)  log(5.2)  log(4.9)  log(6.1)  log(4.2);
          log(0.7)  log(0.8)  log(0.9)  log(3.4)  log(2.1)  log(2.5);
          log(0.7)  log(2.6)  log(1.6)  log(3.6)  log(3.2)  log(2.9);
          log(4.7)  log(2.3)  log(1.6)  log(2.7)  log(1.2)  log(2.5);
          log(1.2)  log(3.6)  log(2.4)  log(4.5)  log(1.6)  log(2.1)]

    # Tij(I,J) = log(processing times)
    Tij = T[log(6.4)  log(4.7)  log(8.3)  log(3.9)  log(2.1)  log(1.2);
             log(6.8)  log(6.4)  log(6.5)  log(4.4)  log(2.3)  log(3.2);
             log(1.0)  log(6.3)  log(5.4)  log(11.9) log(5.7)  log(6.2);
             log(3.2)  log(3.0)  log(3.5)  log(3.3)  log(2.8)  log(3.4);
             log(2.1)  log(2.5)  log(4.2)  log(3.6)  log(3.7)  log(2.2)]

    # Q(I) = production rates
    Q  = T[250000, 150000, 180000, 160000, 120000]

    # ALPHA(J) = 250 for all J, BETA(J) = 0.6 for all J
    ALPHA = T(250)
    BETA  = T(0.6)

    # LOGI(K) = log(K)
    LOGI = T[log(1), log(2), log(3), log(4)]

    # Bounds
    TLO = T[0.729961, 0.530628, 1.09024, -0.133531, 0.0487901]
    TUP = T[2.11626,  1.91626,  2.47654,  1.25276,   1.43508]
    BLO = T[4.45966, 3.74950, 4.49144, 3.14988, 3.04452]
    BUP = T[397.747, 882.353, 833.333, 638.298, 666.667]

    # Variable layout (48 total):
    #   x[1..5]   = TL(5..1) reversed:  TL(i) = x[6-i]
    #   x[6..10]  = B(5..1) reversed:   B(i)  = x[11-i]
    #   x[11..16] = V in order V(5),V(6),V(4),V(3),V(2),V(1): V(j) = x[VPOS[j]]
    #   x[17..22] = N(6..1) reversed:   N(j)  = x[23-j]
    #   x[23..46] = Y(k,j) k-inner, j-outer: Y(k,j) = x[22+(j-1)*4+k]
    #   x[47]     = B(6)  (unused, lvar=0, uvar=Inf)
    #   x[48]     = TL(6) (unused, lvar=0, uvar=Inf)
    VPOS = (16, 15, 14, 13, 11, 12)  # VPOS[j] = position of V(j) for j=1..6

    ExaModels.@add_var(c, x, 48;
        lvar = T[TLO[5], TLO[4], TLO[3], TLO[2], TLO[1],
                 BLO[5], BLO[4], BLO[3], BLO[2], BLO[1],
                 VL, VL, VL, VL, VL, VL,
                 zeros(6)...,
                 zeros(24)...,
                 0, 0],
        uvar = T[TUP[5], TUP[4], TUP[3], TUP[2], TUP[1],
                 BUP[5], BUP[4], BUP[3], BUP[2], BUP[1],
                 VU, VU, VU, VU, VU, VU,
                 fill(LOGNU, 6)...,
                 ones(24)...,
                 Inf, Inf])

    # Objective: COST = sum_j ALPHA * exp(N(j) + BETA*V(j))
    # N(j) = x[23-j], V(j) = x[VPOS[j]]
    ExaModels.@add_obj(c,
        ALPHA * exp(x[22] + BETA * x[16]) +   # j=1: N(1)=x[22], V(1)=x[16]
        ALPHA * exp(x[21] + BETA * x[15]) +   # j=2: N(2)=x[21], V(2)=x[15]
        ALPHA * exp(x[20] + BETA * x[14]) +   # j=3: N(3)=x[20], V(3)=x[14]
        ALPHA * exp(x[19] + BETA * x[13]) +   # j=4: N(4)=x[19], V(4)=x[13]
        ALPHA * exp(x[18] + BETA * x[11]) +   # j=5: N(5)=x[18], V(5)=x[11]
        ALPHA * exp(x[17] + BETA * x[12]))     # j=6: N(6)=x[17], V(6)=x[12]

    # ── Constraints ──────────────────────────────────────────────────────────
    # CUTEst ordering (determined empirically from sifdecoder):
    #
    # cons[1..5]:   SOS1(j=5,4,3,2,1)  E  sum_k Y(k,j) - 1 = 0
    # cons[6..11]:  NPAR(j=6,5,4,3,2,1)  E  sum_k LOGI(k)*Y(k,j) - N(j) = 0
    # cons[12]:     SOS1(j=6)             E
    # cons[13..18]: VOL(i=3, j=1..6)    G  V(j) - B(3) - S(3,j) >= 0
    # cons[19..24]: VOL(i=4, j=1..6)    G
    # cons[25..30]: VOL(i=5, j=1..6)    G
    # cons[31..36]: CYCL(i=1, j=1..6)   G  N(j) + TL(1) - T(1,j) >= 0
    # cons[37..42]: CYCL(i=2, j=1..6)   G
    # cons[43..48]: CYCL(i=3, j=1..6)   G
    # cons[49..54]: CYCL(i=4, j=1..6)   G
    # cons[55..60]: CYCL(i=5, j=1..6)   G
    # cons[61..66]: VOL(i=2, j=6..1)    G  (j reversed!)
    # cons[67..72]: VOL(i=1, j=6..1)    G  (j reversed!)
    # cons[73]:     HORIZON              L  sum_i Q(i)*exp(TL(i)-B(i)) - H <= 0

    # SOS1(j=5,4,3,2,1): sum_k Y(k,j) - 1 = 0
    for j in [5, 4, 3, 2, 1]
        base = 22 + (j - 1) * 4
        ExaModels.@add_con(c, x[base+1] + x[base+2] + x[base+3] + x[base+4] - T(1))
    end

    # NPAR(j=6,5,4,3,2,1): LOGI(1)*Y(1,j) + ... + LOGI(4)*Y(4,j) - N(j) = 0
    # Note LOGI(1)=0 so Y(1,j) term vanishes.
    for j in [6, 5, 4, 3, 2, 1]
        base = 22 + (j - 1) * 4
        nj   = 23 - j   # position of N(j)
        ExaModels.@add_con(c, LOGI[1] * x[base+1] + LOGI[2] * x[base+2] +
                              LOGI[3] * x[base+3] + LOGI[4] * x[base+4] - x[nj])
    end

    # SOS1(j=6)
    let j = 6
        base = 22 + (j - 1) * 4
        ExaModels.@add_con(c, x[base+1] + x[base+2] + x[base+3] + x[base+4] - T(1))
    end

    # VOL(i=3, j=1..6): V(j) - B(3) - S(3,j) >= 0
    # B(3) = x[11-3] = x[8]; V(j) = x[VPOS[j]]
    let bi = 8  # B(3)=x[8]
        for j in 1:6
            rhs = S[3, j]
            ExaModels.@add_con(c, x[VPOS[j]] - x[bi] - rhs; lcon = T(0), ucon = T(Inf))
        end
    end

    # VOL(i=4, j=1..6): V(j) - B(4) - S(4,j) >= 0
    let bi = 7  # B(4)=x[7]
        for j in 1:6
            rhs = S[4, j]
            ExaModels.@add_con(c, x[VPOS[j]] - x[bi] - rhs; lcon = T(0), ucon = T(Inf))
        end
    end

    # VOL(i=5, j=1..6): V(j) - B(5) - S(5,j) >= 0
    let bi = 6  # B(5)=x[6]
        for j in 1:6
            rhs = S[5, j]
            ExaModels.@add_con(c, x[VPOS[j]] - x[bi] - rhs; lcon = T(0), ucon = T(Inf))
        end
    end

    # CYCL(i=1..5, j=1..6): N(j) + TL(i) - T(i,j) >= 0
    # N(j)=x[23-j], TL(i)=x[6-i]
    for i in 1:5
        tli = 6 - i   # position of TL(i)
        for j in 1:6
            nj  = 23 - j  # position of N(j)
            rhs = Tij[i, j]
            ExaModels.@add_con(c, x[nj] + x[tli] - rhs; lcon = T(0), ucon = T(Inf))
        end
    end

    # VOL(i=2, j=6..1): V(j) - B(2) - S(2,j) >= 0  (j reversed!)
    let bi = 9  # B(2)=x[9]
        for j in [6, 5, 4, 3, 2, 1]
            rhs = S[2, j]
            ExaModels.@add_con(c, x[VPOS[j]] - x[bi] - rhs; lcon = T(0), ucon = T(Inf))
        end
    end

    # VOL(i=1, j=6..1): V(j) - B(1) - S(1,j) >= 0  (j reversed!)
    let bi = 10  # B(1)=x[10]
        for j in [6, 5, 4, 3, 2, 1]
            rhs = S[1, j]
            ExaModels.@add_con(c, x[VPOS[j]] - x[bi] - rhs; lcon = T(0), ucon = T(Inf))
        end
    end

    # HORIZON: sum_i Q(i)*exp(TL(i)-B(i)) - H <= 0
    # TL(i)=x[6-i], B(i)=x[11-i]
    ExaModels.@add_con(c,
        Q[1] * exp(x[5]  - x[10]) +   # i=1: TL(1)=x[5],  B(1)=x[10]
        Q[2] * exp(x[4]  - x[9])  +   # i=2: TL(2)=x[4],  B(2)=x[9]
        Q[3] * exp(x[3]  - x[8])  +   # i=3: TL(3)=x[3],  B(3)=x[8]
        Q[4] * exp(x[2]  - x[7])  +   # i=4: TL(4)=x[2],  B(4)=x[7]
        Q[5] * exp(x[1]  - x[6])  - H; # i=5: TL(5)=x[1],  B(5)=x[6]
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
