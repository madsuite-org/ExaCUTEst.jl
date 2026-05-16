# JNLBRNG1: Journal bearing quadratic problem (excentricity=0.1), PT=PY=100
# PT*PY = 10000 variables, bounded below by 0 (interior) or fixed to 0 (boundary)
# Pure quadratic optimization: min linear + quadratic terms
# ISQ element: IV U = V1 - V2, F = U^2 = (V1-V2)^2
# Objective = sum over interior nodes:
#   COEFF*X(I,J) [linear]
#   + LA/HT2 * (X(I+1,J)-X(I,J))^2 [right elements]
#   + MU/HT2 * (X(I-1,J)-X(I,J))^2 [left elements]
#   + LA/HY2 * (X(I,J+1)-X(I,J))^2 [up elements]
#   + MU/HY2 * (X(I,J-1)-X(I,J))^2 [down elements]
# where:
#   EX = 0.1 (excentricity)
#   LT = 8*PI/4 = 2*PI, LY = 20
#   HT = LT/(PT-1), HY = LY/(PY-1)
#   XI(I) = (I-1)*HT for I=1..PT
#   W(I) = (1 + EX*cos(XI(I)))^3
#   For right triangle (I,J): lambda_{I,J} = (2W(I) + W(I+1)) / 6
#     MU/HT2 = mu / HT^2, LA/HT2 = lambda / HT^2 for HT direction
#   For left triangle (I,J): mu_{I,J} = (2W(I) + W(I-1)) / 6
# Linear term COEFF = -EX * sin(XI(I)) * HT * HY (the product CLINC * SXI(I))
#   CLINC = -EX * HT * HY
#
# Variable index: X(I,J) → (I-1)*PY + J (1-indexed)
# Boundary: X(1,J)=X(PT,J)=X(I,1)=X(I,PY)=0 (fixed)
function jnlbrng1_model(PT = 100, PY = 100; T = Float64, backend = nothing, kwargs...)
    EX = T(0.1)
    atan1 = atan(T(1))
    PI4 = atan1  # pi/4
    LT = T(8) * PI4  # 2*pi
    LY = T(20)

    PTm1 = PT - 1
    PYm1 = PY - 1
    HT = LT / T(PTm1)
    HY = LY / T(PYm1)
    HT_HY = HT / HY
    HY_HT = HY / HT
    CLINC = -EX * HT * HY

    nvar = PT * PY
    vidx(i, j) = (i - 1) * PY + j

    lvar = zeros(T, nvar)     # default lvar=0
    uvar = fill(T(Inf), nvar)
    start = zeros(T, nvar)

    # Fix boundary to 0 (already lvar=uvar=0 for boundaries, just fix uvar)
    for j in 1:PY
        uvar[vidx(1, j)] = T(0)
        uvar[vidx(PT, j)] = T(0)
    end
    for i in 2:(PT-1)
        uvar[vidx(i, 1)] = T(0)
        uvar[vidx(i, PY)] = T(0)
    end

    # Precompute per-row quantities for I=2..PT-1
    XI = T[(i - 1) * HT for i in 1:PT]
    W = T[(T(1) + EX * cos(XI[i]))^3 for i in 1:PT]
    SIN_XI = T[sin(XI[i]) for i in 1:PT]

    # For each interior row I (2..PT-1), compute right-group and left-group coefficients
    # Right group (I, J) → elements A(I,J) and B(I,J) with coeff LA/HT2 and LA/HY2
    # Lambda_{I} = (2*W(I) + W(I+1)) / 6
    # LA/HT2 = Lambda * HY/HT
    # LA/HY2 = Lambda * HT/HY
    # Left group (I, J) → elements C(I,J) and D(I,J) with coeff MU/HT2 and MU/HY2
    # Mu_{I} = (2*W(I) + W(I-1)) / 6
    # MU/HT2 = Mu * HY/HT
    # MU/HY2 = Mu * HT/HY

    # For JNLBRNG1:
    # GR(I,J) group: SCALE=2, elements A(I,J) and B(I,J) with LA/HT2 and LA/HY2
    # GL(I,J) group: SCALE=2, elements C(I,J) and D(I,J) with MU/HT2 and MU/HY2
    # G group: linear terms for interior nodes (I=2..PT-1, J=2..PY-1), COEFF=CLINC*sin(XI(I))

    # Group structure (from JNLBRNG1):
    # GR(I,J) for I=1..PT-1, J=1..PY-1: right-upper triangle
    #   SCALE=2: L2 group with scale 2 → contribution = v^2 / 2
    #   But identity (no group type), SCALE=2 → contribution = v / 2? No!
    #   Wait: GR(I,J) has no GROUP TYPE → identity, SCALE=2 → for objective, contribution = v
    #   Actually for identity type, scale is ignored. So contribution = group_value.
    #   Group value GR(I,J) = LA/HT2*(X(I+1,J)-X(I,J))^2 + LA/HY2*(X(I,J+1)-X(I,J))^2
    #   (because A and B elements are added with coefficient LA/HT2 and LA/HY2 respectively)
    #   Wait: GR(I,J) starts with ZN = identity, and SCALE=2. For identity with SCALE sigma:
    #   the objective group contribution = v (scale ignored for identity in objective).
    #   So SCALE=2 on XN (objective) identity group → contribution = group_value.

    # Net objective per interior node (I,J):
    # Linear: G += COEFF*X(I,J)
    # From right triangles GR(I-1,J) and GR(I,J): both contribute (X(I,J+1)-X(I,J))^2 terms
    # This gets complicated. Let me just expand the full quadratic directly.

    # Each interior node (I,J) with I=2..PT-1, J=2..PY-1 has:
    # Linear: CLINC * sin(XI[I]) * X(I,J) (from group G)
    # From right triangles (i belongs to groups GR(I,J) and GR(I-1,J)):
    #   GR(I,J): A(I,J)=ISQ(X(I+1,J),X(I,J)) with coeff LA_I/HT2
    #            B(I,J)=ISQ(X(I,J+1),X(I,J)) with coeff LA_I/HY2
    # From left triangles GL(I,J) and GL(I+1,J) that contain X(I,J):
    #   GL(I,J): C(I,J)=ISQ(X(I-1,J),X(I,J)) with coeff MU_I/HT2
    #            D(I,J)=ISQ(X(I,J-1),X(I,J)) with coeff MU_I/HY2
    # (Each ISQ contribution = (V1-V2)^2 = (difference)^2)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Precompute row data for interior rows I=2..PT-1
    interior_rows = 2:(PT-1)
    COEFF_row = T[CLINC * SIN_XI[i] for i in interior_rows]
    LA_HT2 = T[(T(2) * W[i] + W[i+1]) / T(6) * HY_HT for i in interior_rows]  # lambda * HY/HT
    LA_HY2 = T[(T(2) * W[i] + W[i+1]) / T(6) * HT_HY for i in interior_rows]  # lambda * HT/HY
    MU_HT2 = T[(T(2) * W[i] + W[i-1]) / T(6) * HY_HT for i in interior_rows]  # mu * HY/HT
    MU_HY2 = T[(T(2) * W[i] + W[i-1]) / T(6) * HT_HY for i in interior_rows]  # mu * HT/HY

    ExaModels.@add_par(c, COEFF_R, COEFF_row)
    ExaModels.@add_par(c, LA_T2, LA_HT2)
    ExaModels.@add_par(c, LA_Y2, LA_HY2)
    ExaModels.@add_par(c, MU_T2, MU_HT2)
    ExaModels.@add_par(c, MU_Y2, MU_HY2)

    # ri = i - 1 (1-indexed into interior row arrays, corresponds to row i of grid)
    # Interior nodes: i=2..PT-1 (ri=1..PT-2), j=2..PY-1

    # Linear terms for interior nodes
    ExaModels.@add_obj(c,
        COEFF_R[ri] * x[(ri + 1 - 1) * PY + j]
        for ri in 1:(PT-2) for j in 2:(PY-1))

    # Quadratic terms from right triangles GR(I,J) for I=1..PT-1, J=1..PY-1
    # GR(I,J) contributes LA_I * HY/HT * (X(I+1,J)-X(I,J))^2 + LA_I * HT/HY * (X(I,J+1)-X(I,J))^2
    # But GR is defined for I=1..PT-1 and J=1..PY-1. Only those where all involved nodes
    # are within the domain contribute. For boundary nodes X=0.
    # I=1..PT-1 uses row data from row I (for lambda).
    # Let ri = I-1 (1..PT-2 for interior rows I=2..PT-1)
    # LA at row I (interior): LA_HT2[ri] where ri=I-1

    # Right triangle group GR(I,J) for I=1..PT-1: uses W(I) and W(I+1)
    # Only contributes for interior grid rows I=1..PT-1.
    # Element A(I,J): V1=X(I+1,J), V2=X(I,J) → (X(I+1,J)-X(I,J))^2
    # Element B(I,J): V1=X(I,J+1), V2=X(I,J) → (X(I,J+1)-X(I,J))^2

    # For all I=1..PT-1, J=1..PY-1 (including boundary handled by lvar=uvar=0)
    # LA for row I (right triangle uses rows I and I+1):
    # Lambda_I = (2*W[I] + W[I+1]) / 6
    # LA/HT2[I] = Lambda_I * HY/HT, LA/HY2[I] = Lambda_I * HT/HY

    all_la_ht2 = T[(T(2) * W[i] + W[i+1]) / T(6) * HY_HT for i in 1:(PT-1)]
    all_la_hy2 = T[(T(2) * W[i] + W[i+1]) / T(6) * HT_HY for i in 1:(PT-1)]
    all_mu_ht2 = T[(T(2) * W[i] + W[i-1]) / T(6) * HY_HT for i in 2:PT]
    all_mu_hy2 = T[(T(2) * W[i] + W[i-1]) / T(6) * HT_HY for i in 2:PT]

    ExaModels.@add_par(c, ALL_LA_T2, all_la_ht2)
    ExaModels.@add_par(c, ALL_LA_Y2, all_la_hy2)
    ExaModels.@add_par(c, ALL_MU_T2, all_mu_ht2)
    ExaModels.@add_par(c, ALL_MU_Y2, all_mu_hy2)

    # Right triangle GR(I,J): I=1..PT-1, J=1..PY-1
    ExaModels.@add_obj(c,
        ALL_LA_T2[ri] * (x[ri * PY + j] - x[(ri - 1) * PY + j])^2 +
        ALL_LA_Y2[ri] * (x[(ri - 1) * PY + j + 1] - x[(ri - 1) * PY + j])^2
        for ri in 1:(PT-1) for j in 1:(PY-1))

    # Left triangle GL(I,J): I=2..PT, J=2..PY
    ExaModels.@add_obj(c,
        ALL_MU_T2[ri] * (x[(ri - 2) * PY + j] - x[(ri - 1) * PY + j])^2 +
        ALL_MU_Y2[ri] * (x[(ri - 1) * PY + j - 1] - x[(ri - 1) * PY + j])^2
        for ri in 2:PT for j in 2:PY)

    return ExaModels.ExaModel(c; kwargs...)
end
