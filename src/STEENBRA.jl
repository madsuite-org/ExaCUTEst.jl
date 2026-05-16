# STEENBRA: Separable quadratic multi-commodity network, 432 vars, 108 constraints
# Source: Steenbrink (1974); SIF input by Ph. Toint, 1990
# classification QNR2-AY-432-108
#
# 9-node network, NARC=18 arcs, NTRIPS=12 commodities
# Variables: D(I,K) and R(I,K) for I=1..NTRIPS, K=1..NARC
# Loop order: outer K=1..NARC, inner I=1..NTRIPS
# d(I,K) = 2*(K-1)*12 + 2*(I-1) + 1,  r(I,K) = d(I,K) + 1
#
# Objective (N group OBJ, identity, no group type):
#   For each arc K: ALPHA(K)*FLOW_D + BETA(K)*FLOW_D^2 + ALPHA(K)*FLOW_R + BETA(K)*FLOW_R^2
#   FLOW_D(K) = sum_{I=1}^{12} D(I,K),  FLOW_R(K) = sum_{I=1}^{12} R(I,K)
#   ALPHA(K) = 0.01*COST(K), except ALPHA(4) = 0; BETA(K) = 0.00001*COST(K)
#
# Constraints (E type, 9 nodes × 12 trips = 108 flow conservation equalities)
function steenbra_model(; T = Float64, backend = nothing, kwargs...)
    NARC   = 18
    NTRIPS = 12

    costs = T[35, 40, 30, 100, 15, 55, 100, 25, 60, 35, 55, 15, 40, 60, 25, 30, 50, 50]
    alpha_v = [0.01 * costs[k] for k in 1:NARC]
    alpha_v[4] = T(0)
    beta_v  = [T(0.00001) * costs[k] for k in 1:NARC]

    # Index helpers: d(I,K)=trip I, arc K (1-based Julia)
    didx(I, K) = 2 * (K - 1) * NTRIPS + 2 * (I - 1) + 1
    ridx(I, K) = didx(I, K) + 1

    nv = 2 * NARC * NTRIPS   # 432
    lvar = fill(T(0), nv)
    uvar = fill(T(Inf), nv)
    start = zeros(T, nv)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Objective: one term per arc, each covering all 12 D and 12 R variables
    arc_data = [
        (
         d1  = didx(1,K),  d2  = didx(2,K),  d3  = didx(3,K),
         d4  = didx(4,K),  d5  = didx(5,K),  d6  = didx(6,K),
         d7  = didx(7,K),  d8  = didx(8,K),  d9  = didx(9,K),
         d10 = didx(10,K), d11 = didx(11,K), d12 = didx(12,K),
         r1  = ridx(1,K),  r2  = ridx(2,K),  r3  = ridx(3,K),
         r4  = ridx(4,K),  r5  = ridx(5,K),  r6  = ridx(6,K),
         r7  = ridx(7,K),  r8  = ridx(8,K),  r9  = ridx(9,K),
         r10 = ridx(10,K), r11 = ridx(11,K), r12 = ridx(12,K),
         al  = alpha_v[K],
         be  = beta_v[K],
        )
        for K in 1:NARC
    ]

    # FLOW_D and FLOW_R inlined (ExaModels macros expand expressions)
    ExaModels.@add_obj(c,
        (d.al * (x[d.d1]+x[d.d2]+x[d.d3]+x[d.d4]+x[d.d5]+x[d.d6]+
                 x[d.d7]+x[d.d8]+x[d.d9]+x[d.d10]+x[d.d11]+x[d.d12]) +
         d.be * (x[d.d1]+x[d.d2]+x[d.d3]+x[d.d4]+x[d.d5]+x[d.d6]+
                 x[d.d7]+x[d.d8]+x[d.d9]+x[d.d10]+x[d.d11]+x[d.d12])^2 +
         d.al * (x[d.r1]+x[d.r2]+x[d.r3]+x[d.r4]+x[d.r5]+x[d.r6]+
                 x[d.r7]+x[d.r8]+x[d.r9]+x[d.r10]+x[d.r11]+x[d.r12]) +
         d.be * (x[d.r1]+x[d.r2]+x[d.r3]+x[d.r4]+x[d.r5]+x[d.r6]+
                 x[d.r7]+x[d.r8]+x[d.r9]+x[d.r10]+x[d.r11]+x[d.r12])^2)
        for d in arc_data)

    # Flow conservation constraints for each trip I=1..12
    # RHS: non-zero only at source/destination nodes of each commodity
    rhs = zeros(T, NTRIPS, 9)
    rhs[1,2]=T(-2000); rhs[1,3]=T(2000)   # commodity 1: 2→3
    rhs[2,2]=T(-2000); rhs[2,4]=T(2000)   # commodity 2: 2→4
    rhs[3,2]=T(-1000); rhs[3,5]=T(1000)   # commodity 3: 2→5
    rhs[4,3]=T(-1000); rhs[4,4]=T(1000)   # commodity 4: 3→4
    rhs[5,3]=T(-2000); rhs[5,5]=T(2000)   # commodity 5: 3→5
    rhs[6,4]=T(-1000); rhs[6,5]=T(1000)   # commodity 6: 4→5
    rhs[7,3]=T(-200);  rhs[7,2]=T(200)    # commodity 7: 3→2
    rhs[8,4]=T(-200);  rhs[8,2]=T(200)    # commodity 8: 4→2
    rhs[9,5]=T(-100);  rhs[9,2]=T(100)    # commodity 9: 5→2
    rhs[10,4]=T(-100); rhs[10,3]=T(100)   # commodity 10: 4→3
    rhs[11,5]=T(-200); rhs[11,3]=T(200)   # commodity 11: 5→3
    rhs[12,5]=T(-100); rhs[12,4]=T(100)   # commodity 12: 5→4

    td = [(
           d1=didx(I,1),  r1=ridx(I,1),   d2=didx(I,2),  r2=ridx(I,2),
           d3=didx(I,3),  r3=ridx(I,3),   d4=didx(I,4),  r4=ridx(I,4),
           d5=didx(I,5),  r5=ridx(I,5),   d6=didx(I,6),  r6=ridx(I,6),
           d7=didx(I,7),  r7=ridx(I,7),   d8=didx(I,8),  r8=ridx(I,8),
           d9=didx(I,9),  r9=ridx(I,9),   d10=didx(I,10),r10=ridx(I,10),
           d11=didx(I,11),r11=ridx(I,11), d12=didx(I,12),r12=ridx(I,12),
           d13=didx(I,13),r13=ridx(I,13), d14=didx(I,14),r14=ridx(I,14),
           d15=didx(I,15),r15=ridx(I,15), d16=didx(I,16),r16=ridx(I,16),
           d17=didx(I,17),r17=ridx(I,17), d18=didx(I,18),r18=ridx(I,18),
           b1=rhs[I,1], b2=rhs[I,2], b3=rhs[I,3], b4=rhs[I,4], b5=rhs[I,5],
           b6=rhs[I,6], b7=rhs[I,7], b8=rhs[I,8], b9=rhs[I,9],
          ) for I in 1:NTRIPS]

    # CUTEst order: I-outer, node-inner (N1(1),N2(1),...,N9(1), N1(2),...,N9(12))
    for I in 1:NTRIPS
        d = td[I]
        ExaModels.@add_con(c, -x[d.d1]+x[d.r1] - x[d.d2]+x[d.r2] - x[d.d3]+x[d.r3] - d.b1)
        ExaModels.@add_con(c, x[d.d1]-x[d.r1] - x[d.d4]+x[d.r4] - x[d.d5]+x[d.r5] - x[d.d6]+x[d.r6] - d.b2)
        ExaModels.@add_con(c, x[d.d2]-x[d.r2] - x[d.d7]+x[d.r7] - x[d.d8]+x[d.r8] - x[d.d9]+x[d.r9] - d.b3)
        ExaModels.@add_con(c, x[d.d4]-x[d.r4] - x[d.d10]+x[d.r10] - x[d.d11]+x[d.r11] - x[d.d12]+x[d.r12] - d.b4)
        ExaModels.@add_con(c, x[d.d7]-x[d.r7] - x[d.d13]+x[d.r13] - x[d.d14]+x[d.r14] - x[d.d15]+x[d.r15] - d.b5)
        ExaModels.@add_con(c, x[d.d10]-x[d.r10] + x[d.d13]-x[d.r13] - x[d.d16]+x[d.r16] - d.b6)
        ExaModels.@add_con(c, x[d.d3]-x[d.r3] + x[d.d5]-x[d.r5] + x[d.d8]-x[d.r8] - x[d.d17]+x[d.r17] - d.b7)
        ExaModels.@add_con(c, x[d.d6]-x[d.r6] + x[d.d9]-x[d.r9] + x[d.d11]-x[d.r11] + x[d.d14]-x[d.r14] + x[d.d17]-x[d.r17] - x[d.d18]+x[d.r18] - d.b8)
        ExaModels.@add_con(c, x[d.d12]-x[d.r12] + x[d.d15]-x[d.r15] + x[d.d16]-x[d.r16] + x[d.d18]-x[d.r18] - d.b9)
    end

    return ExaModels.ExaModel(c; kwargs...)
end
