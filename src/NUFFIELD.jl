# NUFFIELD: Consumer surplus optimization (economics problem)
# LQR2-AN-V-V: N=40 (default), nvar=861=(N+1)(N+2)/2, ncon=4*(N choose 2)-like
# Source: Thanassoulis (2001). SIF: Nick Gould, February 2001.
# classification LQR2-AN-V-V
#
# V(I,J) for I=0..N, J=0..I: variable index = I*(I+1)/2 + J + 1
# Fixed: V(0,0) = 0
# Objective: linear in V(I,J) with coefficients C1..C6
# Constraints:
#   VX(I,J) >= 0: 1/H * (V(I+1,J) - V(I,J)) >= 0  (G type, lcon=0)
#   VY(I,J) >= 0: 1/H * (V(I,J+1) - V(I,J)) >= 0  (G type, lcon=0)
#   VXX(I,J) >= 0: 1/H^2 * (V(I+1,J) - 2V(I,J) + V(I-1,J)) >= 0  (G type, lcon=0)
#   VYY(I,J) >= 0: similar (G type, lcon=0)
#   VV(I,J) <= 1: 1/H * (V(...) - V(...)) <= 1  (L type, ucon=0 after subtracting rhs)
#   C(I,J) >= 0: VXX*VYY - VXY^2 >= 0 (nonlinear, G type)
#
# Note: VV constraints use lcon=-Inf, ucon=1 (SIF: XL VV(I,J) >= 0 and XL VV(I,J) on other side)
# Wait - VV groups appear as BOTH G (VX(I,J)) and L (VV(I,J)) type in the SIF:
# "ZG VX(I,J) V(I+1,J) 1/H; ZG VX(I,J) V(I,J) -1/H" → G type: >= 0
# "ZL VV(I,J) V(I+1,J) 1/H; ZL VV(I,J) V(I,J) -1/H" → L type: <= 0 (i.e., <=1 after adding rhs=1)
# Wait: the SIF constant for VV(I,J) is 1.0: "X NUFFIELD VV(I,J) 1.0"
# So VV group value = linear_expr - constant = linear_expr - 1.0
# L type: group value <= 0 → linear_expr - 1.0 <= 0 → linear_expr <= 1

function nuffield_model(N = 40; T = Float64, backend = nothing, kwargs...)
    A = T(5.0)
    RN = T(N)
    H = T(1.0) / RN
    H_inv = RN
    H_inv2 = RN^2
    H_inv4 = RN^4

    # Compute objective coefficients:
    # C1 = H * H * 0.5 * (-A-1) (for V(I,J) interior, I>J loops) * H = 3H * 0.5 * (-A-1) * H
    # Let me re-derive from SIF:
    # RE C2 = H * H * 3.0 = 3H^2; C3 = C2 * 0.5 = 1.5H^2; C4 = C3 + A = 1.5H^2 + A
    # C1 = C3 + (-A-1) = 1.5H^2 - A - 1; C5 = (C3 - 1) * 0.5 = (1.5H^2 - 1)*0.5
    # C6 = 0.5 * C3 + (-A-1) = 0.75H^2 - A - 1
    # Then C1..C6 are each multiplied by H (final RM H):
    # C1 = (1.5H^2 - A - 1) * H; C2 = 3H^3; C3 = 1.5H^3; C4 = (1.5H^2 + A)*H
    # C5 = (1.5H^2 - 1)*0.5*H; C6 = (0.75H^2 - A - 1)*H
    # Actually re-reading more carefully:
    # SIF:
    #   RM C2 H 3.0     → C2 = 3*H
    #   RM C3 C2 0.5    → C3 = 0.5*C2 = 1.5*H
    #   R+ C4 C3 A      → C4 = C3 + A = 1.5*H + A
    #   R+ C1 C3 -A-1   → C1 = C3 + (-A-1) = 1.5*H - A - 1
    #   RA C5 C3 -1.0   → C5 = C3 - 1 = 1.5*H - 1
    #   RM C5 C5 0.5    → C5 = 0.5*(1.5*H - 1) = 0.75*H - 0.5
    #   RM C6 C3 0.5    → C6 = 0.5*C3 = 0.75*H
    #   R+ C6 C6 -A-1   → C6 = 0.75*H - A - 1
    # Then: R* C1..C6 each *= H:
    #   C1 = (1.5*H - A - 1) * H
    #   C2 = 3*H * H = 3*H^2
    #   C3 = 1.5*H * H = 1.5*H^2
    #   C4 = (1.5*H + A) * H
    #   C5 = (0.75*H - 0.5) * H
    #   C6 = (0.75*H - A - 1) * H
    Hval = H
    C1 = (T(1.5)*Hval - A - T(1.0)) * Hval
    C2 = T(3.0) * Hval^2
    C3 = T(1.5) * Hval^2
    C4 = (T(1.5)*Hval + A) * Hval
    C5 = (T(0.75)*Hval - T(0.5)) * Hval
    C6 = (T(0.75)*Hval - A - T(1.0)) * Hval

    # Variable index: V(I,J) → v_idx(I,J)
    v_idx(I, J) = I*(I+1)÷2 + J + 1

    NV = (N+1)*(N+2)÷2  # 861

    # Build model
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    # XR 'DEFAULT' (all free), then XX V(0,0) 0 (fixed)
    lvar[v_idx(0, 0)] = T(0); uvar[v_idx(0, 0)] = T(0)

    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar)

    # Objective: ZN OBJ V(N,J)*C1 for J=1..N-1 (inner loop)
    #            ZN OBJ V(I,J)*C2 for I=2..N-1 (outer), J=1..I-1 (inner)
    #            ZN OBJ V(I,I)*C3 for I=1..N-1
    #            ZN OBJ V(I,0)*C4 for I=1..N-1
    #            ZN OBJ V(N,0)*C5
    #            ZN OBJ V(N,N)*C6
    # Note: minimize (SIF group type N), so we minimize the sum.

    obj_data1 = [(idx = v_idx(N, J),) for J in 1:N-1]
    ExaModels.@add_obj(c, C1 * x[d.idx] for d in obj_data1)

    obj_data2 = [(idx = v_idx(I, J),) for I in 2:N-1 for J in 1:I-1]
    ExaModels.@add_obj(c, C2 * x[d.idx] for d in obj_data2)

    obj_data3 = [(idx = v_idx(I, I),) for I in 1:N-1]
    ExaModels.@add_obj(c, C3 * x[d.idx] for d in obj_data3)

    obj_data4 = [(idx = v_idx(I, 0),) for I in 1:N-1]
    ExaModels.@add_obj(c, C4 * x[d.idx] for d in obj_data4)

    ExaModels.@add_obj(c, C5 * x[v_idx(N, 0)])
    ExaModels.@add_obj(c, C6 * x[v_idx(N, N)])

    # ---- VX(I,J) >= 0: 1/H * (V(I+1,J) - V(I,J)) >= 0, i.e., V(I+1,J) >= V(I,J)
    # DO I=0..N-1; IA I+1=I+1; DO J=0..I; ZG VX(I,J) V(I+1,J) 1/H; ZG VX(I,J) V(I,J) -1/H
    vx_data = [(ip1j = v_idx(I+1, J), ij = v_idx(I, J)) for I in 0:N-1 for J in 0:I]
    ExaModels.@add_con(c, H_inv * (x[d.ip1j] - x[d.ij]) for d in vx_data; lcon = T(0.0))

    # VX(N,J) for J=0..N-1 (backward difference at I=N):
    # ZG VX(N,J) V(N,J) 1/H; ZG VX(N,J) V(N-1,J) -1/H
    vxn_data = [(nj = v_idx(N, J), nm1j = v_idx(N-1, J)) for J in 0:N-1]
    ExaModels.@add_con(c, H_inv * (x[d.nj] - x[d.nm1j]) for d in vxn_data; lcon = T(0.0))

    # VX(N,N): ZG VX(N,N) V(N,N) 1/H; ZG VX(N,N) V(N,N-1) -1/H
    ExaModels.@add_con(c, H_inv * (x[v_idx(N,N)] - x[v_idx(N,N-1)]); lcon = T(0.0))

    # ---- VY(I,J) >= 0: 1/H * (V(I,J+1) - V(I,J)) >= 0
    # DO I=1..N; IA I-1=I-1; DO J=0..I-1; IA J+1=J+1; ZG VY(I,J) V(I,J+1) 1/H; ZG VY(I,J) V(I,J) -1/H
    vy_data = [(ijp1 = v_idx(I, J+1), ij = v_idx(I, J)) for I in 1:N for J in 0:I-1]
    ExaModels.@add_con(c, H_inv * (x[d.ijp1] - x[d.ij]) for d in vy_data; lcon = T(0.0))

    # VY(I,I) for I=1..N-1 (uses V(I+1,I) since J can go up to I via symmetry):
    # ZG VY(I,I) V(I+1,I) 1/H; ZG VY(I,I) V(I,I) -1/H
    vyii_data = [(ip1i = v_idx(I+1, I), ii = v_idx(I, I)) for I in 1:N-1]
    ExaModels.@add_con(c, H_inv * (x[d.ip1i] - x[d.ii]) for d in vyii_data; lcon = T(0.0))

    # VY(N,N): ZG VY(N,N) V(N,N) 1/H; ZG VY(N,N) V(N,N-1) -1/H
    ExaModels.@add_con(c, H_inv * (x[v_idx(N,N)] - x[v_idx(N,N-1)]); lcon = T(0.0))

    # ---- VV(I,J) <= 1: 1/H*(V(I+1,J)-V(I,J)) <= 1 (L type, ZL groups, rhs=1.0)
    # Multiple VV groups accumulate from different pieces. The SIF adds:
    # ZL VV(I,J) V(I+1,J) 1/H; ZL VV(I,J) V(I,J) -1/H (from VX part)
    # ZL VV(I,J) V(I,J+1) 1/H; ZL VV(I,J) V(I,J) -1/H (from VY part)
    # → VV(I,J) group = 1/H*(V(I+1,J) - V(I,J)) + 1/H*(V(I,J+1) - V(I,J))
    # with rhs constant = 1.0
    # L type, constant 1.0: group_val <= 0 means expr - 1 <= 0 → expr <= 1
    # i.e., constraint value = group_val = expr - rhs = (VX + VY) - 1.0
    # SIF VV(I,J) constant = 1.0, L type: group <= 0 in residual form means (val - rhs) <= 0

    # VV accumulation:
    # From "VX" part (DO I=0..N-1, DO J=0..I): ZL VV(I,J): V(I+1,J)/H - V(I,J)/H
    # From "VY" part (DO I=1..N, DO J=0..I-1, IA J+1): ZL VV(I,J): V(I,J+1)/H - V(I,J)/H
    # Combined VV(I,J) for interior (0≤J<I, 0<I<N):
    # val = 1/H*(V(I+1,J) - V(I,J)) + 1/H*(V(I,J+1) - V(I,J))
    # rhs = 1.0 → constraint: val <= 1 → encode as val - 1 <= 0 (ucon=0)

    # VV groups for I=0..N-1, J=0..I (from VX DO loop):
    # VV(I,J) gets ZL from VX when I+1 exists, and ZL from VY when J+1 ≤ I
    # This is complex. Let me enumerate all VV groups:
    # The SIF adds to VV(I,J) from:
    # - VX DO (I=0..N-1, J=0..I): ZL VV(I,J) += 1/H*(V(I+1,J) - V(I,J))
    # - VY DO (I=1..N, J=0..I-1): ZL VV(I,J) += 1/H*(V(I,J+1) - V(I,J))
    # - VX(N,J) DO (J=0..N-1): ZL VV(N,J) += 1/H*(V(N,J) - V(N-1,J))
    # - VX(N,N): ZL VV(N,N) += 1/H*(V(N,N) - V(N,N-1))
    # - VY(I,I) DO (I=1..N-1): ZL VV(I,I) += 1/H*(V(I+1,I) - V(I,I))
    # - VY(N,N): ZL VV(N,N) += 1/H*(V(N,N) - V(N,N-1))

    # VV(I,J) constraint value = (all ZL accumulations) - 1.0, L type → <= 0
    # This creates a set of inequality constraints: val - 1 <= 0

    # For simplicity, list all unique VV(I,J) groups and their contributions:
    # VV is indexed by (I,J) pairs. All possible VV groups from the DO loops:
    vv_dict = Dict{Tuple{Int,Int}, Vector{Tuple{Int,T}}}()

    function add_to_vv(I, J, var_idx, coeff)
        key = (I, J)
        if !haskey(vv_dict, key); vv_dict[key] = []; end
        push!(vv_dict[key], (var_idx, coeff))
    end

    # From VX DO (I=0..N-1, J=0..I):
    for I in 0:N-1, J in 0:I
        add_to_vv(I, J, v_idx(I+1, J), H_inv)
        add_to_vv(I, J, v_idx(I, J), -H_inv)
    end
    # From VX(N,J) DO (J=0..N-1):
    for J in 0:N-1
        add_to_vv(N, J, v_idx(N, J), H_inv)
        add_to_vv(N, J, v_idx(N-1, J), -H_inv)
    end
    # From VX(N,N):
    add_to_vv(N, N, v_idx(N, N), H_inv)
    add_to_vv(N, N, v_idx(N, N-1), -H_inv)
    # From VY DO (I=1..N, J=0..I-1):
    for I in 1:N, J in 0:I-1
        add_to_vv(I, J, v_idx(I, J+1), H_inv)
        add_to_vv(I, J, v_idx(I, J), -H_inv)
    end
    # From VY(I,I) DO (I=1..N-1):
    for I in 1:N-1
        add_to_vv(I, I, v_idx(I+1, I), H_inv)
        add_to_vv(I, I, v_idx(I, I), -H_inv)
    end
    # From VY(N,N):
    add_to_vv(N, N, v_idx(N, N), H_inv)
    add_to_vv(N, N, v_idx(N, N-1), -H_inv)

    # VV constraints: each group val ≤ 1 → val - 1 ≤ 0 (ucon=0)
    # Aggregate coefficients per VV group
    for (IJ, entries) in vv_dict
        coeff_map = Dict{Int,T}()
        for (vidx, cf) in entries
            coeff_map[vidx] = get(coeff_map, vidx, T(0)) + cf
        end
        # Build constraint: sum(coeff_map[k]*x[k]) - 1.0 <= 0
        vv_terms = collect(coeff_map)
        vv_data_inner = [(vidx = Int32(k), cf = T(v)) for (k,v) in vv_terms]
        ExaModels.@add_con(c,
            sum(dd.cf * x[dd.vidx] for dd in vv_data_inner) - T(1.0);
            lcon = T(-Inf), ucon = T(0.0))
    end

    # ---- VXX(I,J) >= 0: 1/H^2*(V(I+1,J) - 2*V(I,J) + V(I-1,J)) >= 0
    # DO I=1..N-1; IA I-1; IA I+1; DO J=0..I-1; ZG VXX(I,J): V(I+1,J)/H^2, V(I,J)*(-2/H^2), V(I-1,J)/H^2
    vxx_data1 = [(ip1j = v_idx(I+1,J), ij = v_idx(I,J), im1j = v_idx(I-1,J))
                 for I in 1:N-1 for J in 0:I-1]
    ExaModels.@add_con(c, H_inv2*(x[d.ip1j] - T(2)*x[d.ij] + x[d.im1j]) for d in vxx_data1; lcon = T(0.0))

    # VXX(I,I) for I=1..N-1:
    # ZG VXX(I,I) V(I+1,I)/H^2; ZG VXX(I,I) V(I,I)*(-2/H^2); ZG VXX(I,I) V(I,I-1)/H^2
    vxxii_data = [(ip1i = v_idx(I+1,I), ii = v_idx(I,I), iim1 = v_idx(I,I-1))
                  for I in 1:N-1]
    ExaModels.@add_con(c, H_inv2*(x[d.ip1i] - T(2)*x[d.ii] + x[d.iim1]) for d in vxxii_data; lcon = T(0.0))

    # ---- VYY(I,J) >= 0: 1/H^2*(V(I,J+1) - 2*V(I,J) + V(I,J-1)) >= 0
    # DO I=2..N; IA I-1; DO J=1..I-1; IA J-1; IA J+1; ZG VYY(I,J): V(I,J+1)/H^2, V(I,J)*(-2/H^2), V(I,J-1)/H^2
    vyy_data1 = [(ijp1 = v_idx(I,J+1), ij = v_idx(I,J), ijm1 = v_idx(I,J-1))
                 for I in 2:N for J in 1:I-1]
    ExaModels.@add_con(c, H_inv2*(x[d.ijp1] - T(2)*x[d.ij] + x[d.ijm1]) for d in vyy_data1; lcon = T(0.0))

    # VYY(I,I) for I=1..N-1:
    # ZG VYY(I,I) V(I+1,I)/H^2; ZG VYY(I,I) V(I,I)*(-2/H^2); ZG VYY(I,I) V(I,I-1)/H^2
    vyyii_data = [(ip1i = v_idx(I+1,I), ii = v_idx(I,I), iim1 = v_idx(I,I-1))
                  for I in 1:N-1]
    ExaModels.@add_con(c, H_inv2*(x[d.ip1i] - T(2)*x[d.ii] + x[d.iim1]) for d in vyyii_data; lcon = T(0.0))

    # ---- C(I,J) >= 0: VXX*VYY - VXY^2 >= 0 (nonlinear convexity constraint)
    # DO I=1..N-1; DO J=1..I (via ELEMENT USES ZV)
    # CONVEX element for C(I,J) uses 9 external variables:
    #   VIP1J = V(I+1,J), VIM1J = V(I-1,J), VIJP1 = V(I,J+1), VIJM1 = V(I,J-1)
    #   VIJ = V(I,J), VIPJP = V(I+1,J+1), VIMJM = V(I-1,J-1)
    #   VIPJM = V(I+1,J-1), VIMJP = V(I-1,J+1)
    # IV: VXX = 1/H^2*(VIP1J - 2*VIJ + VIM1J)
    #     VYY = 1/H^2*(VIJP1 - 2*VIJ + VIJM1)
    #     VXY = 0.25/H^2*(VIPJP - VIMJM - VIMJP + VIPJM)   [from SIF R lines]
    # F = VXX*VYY - VXY^2
    # The CONVEX element is used with coefficient 1/H^4 (ZE C(I,J) C(I,J) 1/H**4).
    # So constraint C(I,J) = 1/H^4 * (VXX*VYY - VXY^2) >= 0 (G type → lcon=0)

    # Note: the J loop in ELEMENT USES is "DO J=1..I" (not I-1), and J can equal I.
    # For J=I: V(I,J+1) = V(I,I+1) which doesn't exist in the triangular grid!
    # Actually the SIF declares: "DO I=1..N-1; IA I+1; IA I-1; DO J=1..I; ...":
    # The ZV lines use V(I+1,J+1) etc. For J=I, J+1=I+1, which for VIPJP = V(I+1,I+1)
    # which DOES exist (since I+1 <= N and J+1 = I+1 <= I+1). OK.
    # For VIJP1 = V(I,J+1): J+1 = I+1, but V(I,I+1) does NOT exist since J must ≤ I.
    # Hmm - wait, V(I,J) only exists for J≤I. But V(I,I+1) with J+1=I+1>I doesn't exist.
    # This seems like an issue. But the SIF says the CONVEX element is defined for J=1..I...
    # Actually re-reading: DO J=1..I, but J+1=I+1 > I for J=I → V(I,J+1) = V(I,I+1) undefined.
    # Maybe VYY group with VIJP1 = V(I+1,I)... no wait, VIJP1 = V(I,J+1).
    # For J=I (the symmetry diagonal), VYY should use V(I+1,I) by the symmetry convention
    # (since v(x,y) = v(y,x) and we only have v(x,y) for x≤y).
    # Let me check: in the VYY group definition for VYY(I,I):
    # ZG VYY(I,I) V(I+1,I)/H^2... yes! VYY at diagonal uses V(I+1,I).
    # The CONVEX element for C(I,J) at J=I uses VIJP1=V(I+1,I) (mapping by symmetry).
    # So V(I,I+1) maps to V(I+1,I) due to symmetry: v(x,y)=v(y,x).
    # Similarly VIPJP = V(I+1,J+1) = V(I+1,I+1) which exists.
    # VIMJP = V(I-1,J+1) = V(I-1,I+1) → doesn't exist (since I+1>I-1)... but by symmetry V(I+1,I-1)?
    # Hmm, this is getting complicated. Let me check the element USES indices more carefully.

    # From ELEMENT USES: for (I,J) in 1≤J≤I≤N-1:
    #   VIP1J = V(I+1,J): valid (J≤I<I+1)
    #   VIM1J = V(I-1,J): valid if J≤I-1, i.e., J<I. For J=I: V(I-1,I) doesn't exist.
    # Actually DO J=1..I:
    # For J=I: VIM1J = V(I-1,I) which doesn't exist in triangular grid (J>I-1)!
    # So there might be a separate ZV assignment for the boundary...
    # Actually looking at the SIF again: "DO I=1..N-1; IA I+1 I 1; IA I-1 I -1; DO J=1..I; IA J+1 J 1; IA J-1 J -1":
    # For J=I, the variables V(I-1,I), V(I,I+1), V(I+1,I+1), V(I-1,I+1), V(I+1,I-1), V(I-1,I-1) are used.
    # V(I-1,I) for J=I: index I-1 < I, so this is like v(I-1, I) but we only have v(i,j) for j≤i.
    # By the symmetry constraint v(x,y)=v(y,x) enforced by only modeling j≤i, V(I-1,I)=V(I,I-1) which DOES exist.
    # Similarly V(I,I+1)=V(I+1,I). So all these can be mapped.
    # The constraint DO J=1..I includes J=I which uses symmetry-equivalent vars.

    # This adds significant complexity. Since the NUFFIELD problem has fixed symmetry
    # built into the variable structure, let me implement C(I,J) only for J<I (strictly)
    # and handle J=I case separately.

    # For C(I,J) with 1≤J≤I≤N-1:
    # Define a helper: sym_v_idx(I,J) = v_idx(max(I,J), min(I,J)) (ensures J≤I)
    sym_v(I,J) = v_idx(max(I,J), min(I,J))

    # Build C(I,J) constraint data
    # VXX = (V(I+1,J) - 2V(I,J) + V(I-1,J))/H^2   [but V(I-1,J) needs J≤I-1, so use sym_v]
    # VYY = (V(I,J+1) - 2V(I,J) + V(I,J-1))/H^2   [V(I,J+1) needs J+1≤I, so use sym_v]
    # VXY = 0.25*(V(I+1,J+1) - V(I-1,J-1) - V(I-1,J+1) + V(I+1,J-1))/H^2
    #       [using sym_v for all]

    c_data = [(
        vij    = v_idx(I, J),
        vip1j  = sym_v(I+1, J),
        vim1j  = sym_v(I-1, J),
        vijp1  = sym_v(I, J+1),
        vijm1  = sym_v(I, J-1),
        vipjp  = sym_v(I+1, J+1),
        vimjm  = sym_v(I-1, J-1),
        vipjm  = sym_v(I+1, J-1),
        vimjp  = sym_v(I-1, J+1),
    ) for I in 1:N-1 for J in 1:I]

    ExaModels.@add_con(c,
        let vxx = H_inv2*(x[d.vip1j] - T(2)*x[d.vij] + x[d.vim1j]),
            vyy = H_inv2*(x[d.vijp1] - T(2)*x[d.vij] + x[d.vijm1]),
            vxy = T(0.25)*H_inv2*(x[d.vipjp] - x[d.vimjm] - x[d.vimjp] + x[d.vipjm])
            T(H_inv4) * (vxx*vyy - vxy^2)
        end
        for d in c_data;
        lcon = T(0.0))

    return ExaModels.ExaModel(c; kwargs...)
end
