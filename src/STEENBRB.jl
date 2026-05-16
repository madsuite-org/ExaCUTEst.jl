# STEENBRB-G: Nonlinear multi-commodity network design problems
# Source: Steenbrink (1974); SIF input by Ph. Toint, June 1990.
#
# 9-node network, NARCS=18 arcs.
# Variants differ in NTRIPS (12 or 14), minimum capacities, and cost coefficients.
#
# Variables per arc K (outer loop K=1..NARCS, inner I=1..NTRIPS):
#   CD(K): design capacity for direct flow, index (K-1)*(2+2*NTRIPS)+1
#   CR(K): design capacity for reverse flow, index (K-1)*(2+2*NTRIPS)+2
#   D(I,K): direct flow commodity I on arc K, index (K-1)*stride+2+2*(I-1)+1
#   R(I,K): reverse flow commodity I on arc K, index (K-1)*stride+2+2*(I-1)+2
#
# Objective (N group, identity type):
#   For each arc K:
#     XT element (D): F = LT(K)*FLOW_D + LC(K)*FLOW_D^3/CD(K)^2
#     XT element (R): F = LT(K)*FLOW_R + LC(K)*FLOW_R^3/CR(K)^2
#     IIJ element (D): LA(K)*sqrt(CD(K) - CMIN_D(K))
#     IIJ element (R): LA(K)*sqrt(CR(K) - CMIN_R(K))
#   where LT(K)=COST(K)*TZERO, LC(K)=COST(K)*CCR, LA(K)=COST(K)*ALPH
#   ALPH=TZERO=CCR=0.01; CMIN(K)=CMD(K)-NONZ, NONZ=0.01
#
# Equality constraints (flow conservation, 9 nodes × NTRIPS trips)

function _steenbrb_model(NTRIPS::Int, cmd4_base, cmr4_base, la4_mult;
                         T = Float64, backend = nothing, kwargs...)
    NARCS  = 18
    ALPH   = T(0.01)
    TZERO  = T(0.01)
    CCR    = T(0.01)
    NONZ   = T(0.01)

    costs = T[35, 40, 30, 100, 15, 55, 100, 25, 60, 35, 55, 15, 40, 60, 25, 30, 50, 50]

    lt  = [costs[k] * TZERO for k in 1:NARCS]
    lc  = [costs[k] * CCR   for k in 1:NARCS]
    la  = [costs[k] * ALPH  for k in 1:NARCS]
    la[4] = la[4] * T(la4_mult)

    # Minimum capacities: default CMD(K)=CMR(K)=0+NONZ.
    # For arc 4 overrides: AE CMD(4) sets it directly (no +NONZ), then
    # SHIFT = CMD(4)+MNONZ = CMD(4)-NONZ in IIJ element.
    # cmd4_base=0 means no override → CMD(4)=NONZ; cmd4_base>0 means CMD(4)=cmd4_base exactly.
    cmd = fill(NONZ, NARCS)
    cmr = fill(NONZ, NARCS)
    if cmd4_base > 0
        cmd[4] = T(cmd4_base)
    end
    if cmr4_base > 0
        cmr[4] = T(cmr4_base)
    end

    # CMIN for IIJ: SHIFT = CMD(I) + MNONZ = CMD(I) - NONZ
    cmin_d = [cmd[k] - NONZ for k in 1:NARCS]
    cmin_r = [cmr[k] - NONZ for k in 1:NARCS]

    stride = 2 + 2 * NTRIPS
    nv     = NARCS * stride

    lvar  = Vector{T}(undef, nv)
    uvar  = Vector{T}(undef, nv)
    start = Vector{T}(undef, nv)
    for K in 1:NARCS
        base = (K - 1) * stride
        lvar[base+1]  = cmd[K]; uvar[base+1] = T(Inf); start[base+1] = T(0.1)  # CD(K)
        lvar[base+2]  = cmr[K]; uvar[base+2] = T(Inf); start[base+2] = T(0.1)  # CR(K)
        for I in 1:NTRIPS
            di = base + 2 + 2*(I-1) + 1
            ri = base + 2 + 2*(I-1) + 2
            lvar[di] = T(0); uvar[di] = T(Inf); start[di] = T(0.1)
            lvar[ri] = T(0); uvar[ri] = T(Inf); start[ri] = T(0.1)
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    cd_idx(K)    = (K - 1) * stride + 1
    cr_idx(K)    = (K - 1) * stride + 2
    d_idx(I, K)  = (K - 1) * stride + 2 + 2*(I-1) + 1
    r_idx(I, K)  = (K - 1) * stride + 2 + 2*(I-1) + 2

    # Build objective data: one record per arc with all flow variable indices
    if NTRIPS == 12
        arc_data12 = [
            (
             cd=cd_idx(K), cr=cr_idx(K),
             d1=d_idx(1,K),  d2=d_idx(2,K),  d3=d_idx(3,K),  d4=d_idx(4,K),
             d5=d_idx(5,K),  d6=d_idx(6,K),  d7=d_idx(7,K),  d8=d_idx(8,K),
             d9=d_idx(9,K),  d10=d_idx(10,K),d11=d_idx(11,K),d12=d_idx(12,K),
             r1=r_idx(1,K),  r2=r_idx(2,K),  r3=r_idx(3,K),  r4=r_idx(4,K),
             r5=r_idx(5,K),  r6=r_idx(6,K),  r7=r_idx(7,K),  r8=r_idx(8,K),
             r9=r_idx(9,K),  r10=r_idx(10,K),r11=r_idx(11,K),r12=r_idx(12,K),
             lt=lt[K], lc=lc[K], la=la[K],
             cmin_d=cmin_d[K], cmin_r=cmin_r[K],
            ) for K in 1:NARCS
        ]
        ExaModels.@add_obj(c,
            let fd = x[d.d1]+x[d.d2]+x[d.d3]+x[d.d4]+x[d.d5]+x[d.d6]+
                     x[d.d7]+x[d.d8]+x[d.d9]+x[d.d10]+x[d.d11]+x[d.d12],
                fr = x[d.r1]+x[d.r2]+x[d.r3]+x[d.r4]+x[d.r5]+x[d.r6]+
                     x[d.r7]+x[d.r8]+x[d.r9]+x[d.r10]+x[d.r11]+x[d.r12]
                d.lt*fd + d.lc*fd^3/x[d.cd]^2 +
                d.lt*fr + d.lc*fr^3/x[d.cr]^2 +
                d.la*sqrt(x[d.cd] - d.cmin_d) +
                d.la*sqrt(x[d.cr] - d.cmin_r)
            end
            for d in arc_data12)
    else # NTRIPS == 14
        arc_data14 = [
            (
             cd=cd_idx(K), cr=cr_idx(K),
             d1=d_idx(1,K),   d2=d_idx(2,K),  d3=d_idx(3,K),  d4=d_idx(4,K),
             d5=d_idx(5,K),   d6=d_idx(6,K),  d7=d_idx(7,K),  d8=d_idx(8,K),
             d9=d_idx(9,K),   d10=d_idx(10,K),d11=d_idx(11,K),d12=d_idx(12,K),
             d13=d_idx(13,K), d14=d_idx(14,K),
             r1=r_idx(1,K),   r2=r_idx(2,K),  r3=r_idx(3,K),  r4=r_idx(4,K),
             r5=r_idx(5,K),   r6=r_idx(6,K),  r7=r_idx(7,K),  r8=r_idx(8,K),
             r9=r_idx(9,K),   r10=r_idx(10,K),r11=r_idx(11,K),r12=r_idx(12,K),
             r13=r_idx(13,K), r14=r_idx(14,K),
             lt=lt[K], lc=lc[K], la=la[K],
             cmin_d=cmin_d[K], cmin_r=cmin_r[K],
            ) for K in 1:NARCS
        ]
        ExaModels.@add_obj(c,
            let fd = x[d.d1]+x[d.d2]+x[d.d3]+x[d.d4]+x[d.d5]+x[d.d6]+
                     x[d.d7]+x[d.d8]+x[d.d9]+x[d.d10]+x[d.d11]+x[d.d12]+
                     x[d.d13]+x[d.d14],
                fr = x[d.r1]+x[d.r2]+x[d.r3]+x[d.r4]+x[d.r5]+x[d.r6]+
                     x[d.r7]+x[d.r8]+x[d.r9]+x[d.r10]+x[d.r11]+x[d.r12]+
                     x[d.r13]+x[d.r14]
                d.lt*fd + d.lc*fd^3/x[d.cd]^2 +
                d.lt*fr + d.lc*fr^3/x[d.cr]^2 +
                d.la*sqrt(x[d.cd] - d.cmin_d) +
                d.la*sqrt(x[d.cr] - d.cmin_r)
            end
            for d in arc_data14)
    end

    # RHS for flow conservation constraints
    rhs = zeros(T, NTRIPS, 9)
    rhs[1,2]=T(-2000); rhs[1,3]=T(2000)
    rhs[2,2]=T(-2000); rhs[2,4]=T(2000)
    rhs[3,2]=T(-1000); rhs[3,5]=T(1000)
    rhs[4,3]=T(-1000); rhs[4,4]=T(1000)
    rhs[5,3]=T(-2000); rhs[5,5]=T(2000)
    rhs[6,4]=T(-1000); rhs[6,5]=T(1000)
    rhs[7,3]=T(-200);  rhs[7,2]=T(200)
    rhs[8,4]=T(-200);  rhs[8,2]=T(200)
    rhs[9,5]=T(-100);  rhs[9,2]=T(100)
    rhs[10,4]=T(-100); rhs[10,3]=T(100)
    rhs[11,5]=T(-200); rhs[11,3]=T(200)
    rhs[12,5]=T(-100); rhs[12,4]=T(100)
    if NTRIPS >= 14
        rhs[13,1]=T(-10000); rhs[13,6]=T(10000)
        rhs[14,6]=T(-1000);  rhs[14,1]=T(1000)
    end

    td = [
        (
         d1=d_idx(I,1),  r1=r_idx(I,1),   d2=d_idx(I,2),  r2=r_idx(I,2),
         d3=d_idx(I,3),  r3=r_idx(I,3),   d4=d_idx(I,4),  r4=r_idx(I,4),
         d5=d_idx(I,5),  r5=r_idx(I,5),   d6=d_idx(I,6),  r6=r_idx(I,6),
         d7=d_idx(I,7),  r7=r_idx(I,7),   d8=d_idx(I,8),  r8=r_idx(I,8),
         d9=d_idx(I,9),  r9=r_idx(I,9),   d10=d_idx(I,10),r10=r_idx(I,10),
         d11=d_idx(I,11),r11=r_idx(I,11), d12=d_idx(I,12),r12=r_idx(I,12),
         d13=d_idx(I,13),r13=r_idx(I,13), d14=d_idx(I,14),r14=r_idx(I,14),
         d15=d_idx(I,15),r15=r_idx(I,15), d16=d_idx(I,16),r16=r_idx(I,16),
         d17=d_idx(I,17),r17=r_idx(I,17), d18=d_idx(I,18),r18=r_idx(I,18),
         b1=rhs[I,1], b2=rhs[I,2], b3=rhs[I,3], b4=rhs[I,4], b5=rhs[I,5],
         b6=rhs[I,6], b7=rhs[I,7], b8=rhs[I,8], b9=rhs[I,9],
        ) for I in 1:NTRIPS
    ]

    # N1(I): arcs 1,2,3 outgoing
    ExaModels.@add_con(c,
        -x[d.d1]+x[d.r1] - x[d.d2]+x[d.r2] - x[d.d3]+x[d.r3] - d.b1
        for d in td)

    # N2(I): arcs 4,5,6 outgoing; arc 1 incoming
    ExaModels.@add_con(c,
        x[d.d1]-x[d.r1] - x[d.d4]+x[d.r4] - x[d.d5]+x[d.r5] - x[d.d6]+x[d.r6] - d.b2
        for d in td)

    # N3(I): arcs 7,8,9 outgoing; arc 2 incoming
    ExaModels.@add_con(c,
        x[d.d2]-x[d.r2] - x[d.d7]+x[d.r7] - x[d.d8]+x[d.r8] - x[d.d9]+x[d.r9] - d.b3
        for d in td)

    # N4(I): arcs 10,11,12 outgoing; arc 4 incoming
    ExaModels.@add_con(c,
        x[d.d4]-x[d.r4] - x[d.d10]+x[d.r10] - x[d.d11]+x[d.r11] - x[d.d12]+x[d.r12] - d.b4
        for d in td)

    # N5(I): arcs 13,14,15 outgoing; arc 7 incoming
    ExaModels.@add_con(c,
        x[d.d7]-x[d.r7] - x[d.d13]+x[d.r13] - x[d.d14]+x[d.r14] - x[d.d15]+x[d.r15] - d.b5
        for d in td)

    # N6(I): arc 16 outgoing; arcs 10,13 incoming
    ExaModels.@add_con(c,
        x[d.d10]-x[d.r10] + x[d.d13]-x[d.r13] - x[d.d16]+x[d.r16] - d.b6
        for d in td)

    # N7(I): arc 17 outgoing; arcs 3,5,8 incoming
    ExaModels.@add_con(c,
        x[d.d3]-x[d.r3] + x[d.d5]-x[d.r5] + x[d.d8]-x[d.r8] - x[d.d17]+x[d.r17] - d.b7
        for d in td)

    # N8(I): arc 18 outgoing; arcs 6,9,11,14,17 incoming
    ExaModels.@add_con(c,
        x[d.d6]-x[d.r6] + x[d.d9]-x[d.r9] + x[d.d11]-x[d.r11] +
        x[d.d14]-x[d.r14] + x[d.d17]-x[d.r17] - x[d.d18]+x[d.r18] - d.b8
        for d in td)

    # N9(I): arcs 12,15,16,18 incoming
    ExaModels.@add_con(c,
        x[d.d12]-x[d.r12] + x[d.d15]-x[d.r15] + x[d.d16]-x[d.r16] + x[d.d18]-x[d.r18] - d.b9
        for d in td)

    return ExaModels.ExaModel(c; kwargs...)
end

# STEENBRB: NTRIPS=12, default min caps (CMD/CMR=NONZ), LA unchanged
function steenbrb_model(; T = Float64, backend = nothing, kwargs...)
    _steenbrb_model(12, 0, 0, 1; T = T, backend = backend, kwargs...)
end

# STEENBRC: NTRIPS=14, default min caps, LA unchanged
function steenbrc_model(; T = Float64, backend = nothing, kwargs...)
    _steenbrb_model(14, 0, 0, 1; T = T, backend = backend, kwargs...)
end

# STEENBRD: NTRIPS=12, CMD(4)=CMR(4)=2000 (SIF AE override), LA unchanged
function steenbrd_model(; T = Float64, backend = nothing, kwargs...)
    _steenbrb_model(12, 2000, 2000, 1; T = T, backend = backend, kwargs...)
end

# STEENBRE: NTRIPS=14, CMD(4)=CMR(4)=2000, LA unchanged
function steenbre_model(; T = Float64, backend = nothing, kwargs...)
    _steenbrb_model(14, 2000, 2000, 1; T = T, backend = backend, kwargs...)
end

# STEENBRF: NTRIPS=12, default min caps, LA(4) halved (AM LA(4) LA(4) 0.5)
function steenbrf_model(; T = Float64, backend = nothing, kwargs...)
    _steenbrb_model(12, 0, 0, 0.5; T = T, backend = backend, kwargs...)
end

# STEENBRG: NTRIPS=14, default min caps, LA(4) halved
function steenbrg_model(; T = Float64, backend = nothing, kwargs...)
    _steenbrb_model(14, 0, 0, 0.5; T = T, backend = backend, kwargs...)
end
