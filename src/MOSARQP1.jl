# MOSARQP1: Convex QP with Laplacian constraints (half active at solution)
# QLR2-AN-V-V: N=2500 variables, M=700 inequality (G type, ≥ rhs) constraints
# Source: Morales-Perez & Sargent, 1991. SIF input: Ph. Toint, August 1993.
# Default: N=2500, M=700, COND=3.0; RTN=floor(sqrt(N+0.1))=50
#
# Objective: 0.5*x'*H*x + c'*x where H = diag(D) + rank-1 correction via Y vector
# Constraints: first M rows of 5-point Laplacian >= rhs (XG groups)
function mosarqp1_model(; T = Float64, backend = nothing, kwargs...)
    N = 2500
    M = 700
    COND = T(3.0)
    RTN = 50  # floor(sqrt(2500 + 0.1)) = 50

    NNZ = 10
    Y_vals = T[-0.3569732, 0.9871576, 0.5619363, -0.1984624, 0.4653328,
                0.7364367, -0.4560378, -0.6457813, -0.0601357, 0.1035624]
    NZ_fracs = T[0.68971452, 0.13452678, 0.51234678, 0.76591423, 0.20857854,
                  0.85672348, 0.04356789, 0.44692743, 0.30136413, 0.91367489]

    # K(I) = floor(NZ(I)*N + 1.1) — SIF uses AA which adds 1.1 before truncation
    K = Int32[floor(Int32, NZ_fracs[i] * N + T(1.1)) for i in 1:NNZ]

    # D(I) = exp((I-1)/(N-1) * COND)
    D = [exp(T(i - 1) / T(N - 1) * COND) for i in 1:N]

    # DY(I) = Y(I) * D(K(I))
    DY = [Y_vals[i] * D[K[i]] for i in 1:NNZ]

    # YN2 = ||Y||^2
    YN2 = sum(y^2 for y in Y_vals)
    neg2_yn2 = T(-2.0) / YN2
    pos4_yn4 = T(4.0) / YN2^2

    # Quadratic center XC: alternating -1,1 pairs
    XC = zeros(T, N)
    i = 1
    while i <= N - 1
        XC[i] = T(-1.0)
        XC[i+1] = T(1.0)
        i += 2
    end
    XC[N] = T(1.0)

    YDY = sum(DY[i] * Y_vals[i] for i in 1:NNZ)
    YXC = sum(Y_vals[i] * XC[K[i]] for i in 1:NNZ)
    YDXC = sum(DY[i] * XC[K[i]] for i in 1:NNZ)

    AA = neg2_yn2 * YXC
    DD = pos4_yn4 * YDY
    BB_CC = DD * YXC + neg2_yn2 * YDXC
    DD_half = DD * T(0.5)

    # Gradient: C(I) = D(I)*XC(I), then add rank-1 corrections
    C_grad = [D[i] * XC[i] for i in 1:N]
    for i in 1:NNZ
        ki = K[i]
        C_grad[ki] += DY[i] * AA + Y_vals[i] * BB_CC
    end

    # Diagonal Hessian: H_diag(I) = D(I) (standard) + rank-1 diagonal terms
    H_diag = copy(D)
    for i in 1:NNZ
        ki = K[i]
        wii = DY[i] * Y_vals[i] * neg2_yn2 + Y_vals[i]^2 * DD_half
        H_diag[ki] += 2 * wii
    end

    # Off-diagonal pairs for objective: (ki, kj, weight)
    off_pairs = [(i = Int32(K[i]), j = Int32(K[jj]),
                  w = T(DY[i]*Y_vals[jj]*neg2_yn2 + DY[jj]*Y_vals[i]*neg2_yn2 + Y_vals[i]*Y_vals[jj]*DD))
                 for i in 1:NNZ for jj in 1:i-1]

    # Build 5-point Laplacian constraints (first M rows of RTN×RTN grid)
    # Grid: point index idx → grid_col = mod(idx-1, RTN)+1, grid_row = div(idx-1, RTN)+1
    # Lap row: center coeff +4; neighbors coeff -1 (if they exist); else coeff 0 on dummy var
    # Since there's no dummy, use 0 coefficient directly by separating cases.
    # Store: center index + at most 4 neighbor (index, coeff) pairs
    # RHS = (4 - num_neighbors - 1) * 0.5 = (3 - num_neighbors) * 0.5
    # Because: A*x0 - p with x0=p=0.5*e, so A_i * x0 - p_i = row_sum*0.5 - 0.5 = (row_sum-1)*0.5
    # row_sum = 4 - num_neighbors → rhs = (3 - num_neighbors)*0.5

    # Separate constraints by number of neighbors (5 cases: 2,3,4 neighbors in practice)
    # Corner has 2 neighbors, edge has 3, interior has 4.
    # For the first M=700 rows of a 50×50 grid:
    # Row 1: top-left corner → grid_col=1, grid_row=1 → 2 neighbors (right, down)
    # Rows 2..49: top edge → 3 neighbors
    # Row 50: top-right corner → 2 neighbors
    # Rows 51..700: interior or next row...

    # Build structured data with separate coefficient storage
    # For ExaModels, use named tuple arrays with full 5-entry lists where missing = 0
    # Use the trick: store all 5 (center+4 neighbors), with coeff=0 for missing neighbors
    # and use var index 1 (any valid index) for missing - multiplied by 0 = no contribution.
    # This avoids conditionals in the kernel.

    con_data = Vector{NamedTuple{(:ci,:il,:ir,:iu,:id,:cl,:cr,:cu,:cd,:rhs),
                                  NTuple{10,T}}}(undef, M)
    for idx in 1:M
        gc = T(mod(idx - 1, RTN) + 1)
        gr = T(div(idx - 1, RTN) + 1)
        have_l = gc > 1
        have_r = gc < RTN
        have_u = gr > 1
        have_d = gr < RTN
        cl = have_l ? T(-1.0) : T(0.0)
        cr = have_r ? T(-1.0) : T(0.0)
        cu = have_u ? T(-1.0) : T(0.0)
        cd = have_d ? T(-1.0) : T(0.0)
        il = have_l ? T(idx - 1) : T(1.0)
        ir = have_r ? T(idx + 1) : T(1.0)
        iu = have_u ? T(idx - RTN) : T(1.0)
        id = have_d ? T(idx + RTN) : T(1.0)
        nneigh = (have_l ? 1 : 0) + (have_r ? 1 : 0) + (have_u ? 1 : 0) + (have_d ? 1 : 0)
        rhs_val = T(3 - nneigh) * T(0.5)
        con_data[idx] = (
            ci = T(idx), il = il, ir = ir, iu = iu, id = id,
            cl = cl, cr = cr, cu = cu, cd = cd,
            rhs = rhs_val
        )
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = zeros(T, N), uvar = fill(T(Inf), N), start = fill(T(0.5), N))

    # Objective diagonal terms: 0.5 * H_diag[i] * x[i]^2
    hd_data = [(i = Int32(i), d = T(H_diag[i])) for i in 1:N]
    ExaModels.@add_obj(c, T(0.5) * d.d * x[d.i]^2 for d in hd_data)

    # Objective linear (gradient)
    cg_data = [(i = Int32(i), g = T(C_grad[i])) for i in 1:N]
    ExaModels.@add_obj(c, d.g * x[d.i] for d in cg_data)

    # Objective off-diagonal (rank-1 correction)
    if !isempty(off_pairs)
        ExaModels.@add_obj(c, d.w * x[d.i] * x[d.j] for d in off_pairs)
    end

    # Constraints (G type: >= rhs → encode as con - rhs >= 0, i.e., lcon=0, ucon=Inf)
    # Using indices as Int (converted from T) - but ExaModels needs Int indexing.
    # Use Int32 arrays instead:
    con_data2 = [(ci = Int32(idx), il = con_data[idx].il, ir = con_data[idx].ir,
                   iu = con_data[idx].iu, id = con_data[idx].id,
                   cl = con_data[idx].cl, cr = con_data[idx].cr,
                   cu = con_data[idx].cu, cd = con_data[idx].cd,
                   rhs = con_data[idx].rhs,
                   ali = Int32(round(con_data[idx].il)), ari = Int32(round(con_data[idx].ir)),
                   aui = Int32(round(con_data[idx].iu)), adi = Int32(round(con_data[idx].id)))
                  for idx in 1:M]

    ExaModels.@add_con(c,
        T(4) * x[d.ci] + d.cl * x[d.ali] + d.cr * x[d.ari] + d.cu * x[d.aui] + d.cd * x[d.adi] - d.rhs
        for d in con_data2;
        lcon = T(0.0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
