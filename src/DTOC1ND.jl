# DTOC1ND: Discrete-time optimal control problem 1 (no-derivative variant)
# N=1000 time periods, NX=2 controls, NY=4 states
# Variables: X(T,I) for T=1..N-1, I=1..NX; Y(T,I) for T=1..N, I=1..NY
# Objective: sum_{T,I} (X(T,I) - 0.5)^4 + sum_{T,I} (Y(T,I) - 0.25)^4
# Constraints: TT(T,J) = Y(T+1,J) - [linear(Y(T,:)) + B(J,:)*X(T,:) + nonlinear(C)] = 0
# Element PR: MUC * X * Y (with parameters C(I,J))
# Group type L4: GVAR^4
# SIF: MU=1, NX=2, NY=4, MU/RXY = 1/(NX+NY) = 1/6
function dtoc1nd_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    NX = 2
    NY = 4
    NXY = NX + NY  # 6

    # Transition matrices B and C
    # B(I,J) = (I-J)/RXY  for I=1..NY, J=1..NX
    # C(I,J) = (I+J)*MU/RXY  for I=1..NY, J=1..NX
    MU = T(1.0)
    RXY = T(NXY)
    B = [T(I - J) / RXY for I in 1:NY, J in 1:NX]
    C = [T(I + J) * MU / RXY for I in 1:NY, J in 1:NX]

    # Variable indexing:
    # X(T,I): index = (T-1)*NX + I, for T=1..N-1, I=1..NX
    # Y(T,I): index = (N-1)*NX + (T-1)*NY + I, for T=1..N, I=1..NY
    NX_total = (N - 1) * NX
    NY_total = N * NY
    NV = NX_total + NY_total

    x_idx(t, i) = (t - 1) * NX + i
    y_idx(t, i) = NX_total + (t - 1) * NY + i

    # Bounds: all free except Y(1,I) = 0 fixed
    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    for i in 1:NY
        lvar[y_idx(1, i)] = T(0.0)
        uvar[y_idx(1, i)] = T(0.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar)

    # Objective: L4 group type: v^4 where v = (X(T,I) - (-0.5)) = X(T,I) + 0.5
    # OX(T,I) has linear X(T,I) with coeff 1, constant rhs=-(-0.5)=0.5 → v = X(T,I) - (-0.5)
    # SIF CONSTANTS: X DTOC1ND OX(T,I) -0.5 means rhs=0.5 for OX groups
    # So v = X(T,I) - 0.5, group type L4: v^4
    obj_x_data = [(xi = x_idx(t, i),) for t in 1:(N-1), i in 1:NX]
    ExaModels.@add_obj(c, (x[d.xi] - T(0.5))^4 for d in obj_x_data)

    # OY(T,I): rhs = 0.25, so v = Y(T,I) - 0.25
    obj_y_data = [(yi = y_idx(t, i),) for t in 1:N, i in 1:NY]
    ExaModels.@add_obj(c, (x[d.yi] - T(0.25))^4 for d in obj_y_data)

    # Constraints TT(T,J) for T=1..N-1, J=1..NY
    # TT(T,J): Y(T+1,J) - 1 - linear(Y(T,:)) - sum_I B(J,I)*X(T,I) - sum_K E(T,K) = 0
    # where E(T,K) is in TT(T,J) for all J (see GROUP USES: DO J DO K XE TT(T,J) E(T,K))
    # Element E(T,K): PR with Y=Y(T, I(K)), X=X(T, J(K)), param MUC=C(I,J)
    # K = 0..NYNX-1, NYNX=NX*NY=8
    # I = K/NX + 1 (1-based), J = K mod NX + 1 (1-based)
    # TT(T,J) = -Y(T+1,J) + Y(T,J)*0.5 + ... + sum_K MUC(K)*Y(T,I(K))*X(T,J(K)) = 0
    # First let's decode:
    # TT(T,1): -Y(T+1,1) + Y(T,1)*0.5 + Y(T,2)*0.25 + B(1,1)*X(T,1) + B(1,2)*X(T,2) + sum_K E(T,K) = 0
    # TT(T,J) for 2<=J<=NY-1: -Y(T+1,J) + Y(T,J)*0.5 + Y(T,J-1)*(-0.25) + Y(T,J+1)*0.25 + B(J,:)*X(T,:) + sum_K E(T,K) = 0
    # TT(T,NY): -Y(T+1,NY) + Y(T,NY)*0.5 + Y(T,NY-1)*(-0.25) + B(NY,:)*X(T,:) + sum_K E(T,K) = 0
    # sum_K E(T,K) = sum_{I=1..NY,J_=1..NX} C(I,J_)*Y(T,I)*X(T,J_)

    # For constraint J at time T, the formula is:
    # -Y(T+1,J) + (linear part in Y(T,:)) + B(J,:)*X(T,:) + sum_K C(I(K),J_(K))*Y(T,I(K))*X(T,J_(K)) = 0
    # The "each constraint gets all elements" means ALL elements E(T,K) appear in ALL constraints TT(T,J).
    # But that gives a nonseparable structure. We need to combine them.

    # Actually, re-reading the SIF more carefully:
    # In GROUP USES: DO T DO J DO K XE TT(T,J) E(T,K)
    # This means E(T,K) contributes to ALL J constraints at time T, for each K.
    # So constraint TT(T,J) = linear_part_J(Y,X) + sum_K C(I(K),J_(K))*Y(T,I(K))*X(T,J_(K)) = 0
    # The sum is the same for all J! This seems odd but is what the SIF says.
    # Let's encode: TT(T,J) = f_J(Y(T,:), X(T,:), Y(T+1,J)) + S(T) = 0
    # where S(T) = sum_{I,J_} C(I,J_)*Y(T,I)*X(T,J_)

    # Build constraint data per (T, J) pair
    # Linear part for constraint J at time T:
    #   J=1: Y(T,1)*0.5 + Y(T,2)*0.25 + B(1,:)*X(T,:) - Y(T+1,1)
    #   2<=J<=NY-1: Y(T,J)*0.5 + Y(T,J-1)*(-0.25) + Y(T,J+1)*0.25 + B(J,:)*X(T,:) - Y(T+1,J)
    #   J=NY: Y(T,NY)*0.5 + Y(T,NY-1)*(-0.25) + B(NY,:)*X(T,:) - Y(T+1,NY)

    # Nonlinear part: sum_{I=1..NY, Jx=1..NX} C(I,Jx)*Y(T,I)*X(T,Jx) added to each constraint J
    # For ExaModels we cannot share the nonlinear sum across constraints; we need to add it to each.
    # Since all J constraints at time T share the same nonlinear sum, we add it once per (T,J).

    # We'll flatten: for each T, J, encode the full constraint including the nonlinear product terms.
    # This gives N*NY*NX*NY = 1000*4*2*4 = 32000 nonlinear terms across all constraints.
    # Actually we need NX*NY=8 product terms per constraint, and there are (N-1)*NY=3996 constraints.

    # Per constraint (T,J), nonlinear terms: sum_{I,Jx} C(I,Jx)*Y(T,I)*X(T,Jx)
    # We'll build a data array of (yI_idx, xJ_idx, CIJ) tuples for each (T,J)
    # But since the nonlinear part is the same for all J at given T, we can add it as a separate objective/constraint term.
    # Actually we MUST include it in each constraint. Use structured data.

    # Build a 2D data array: con_data[(T-1)*NY+J] = (yTP1J, yTJ, yTJm1, yTJp1, xT1, xT2, BJx1, BJx2, coeff_yTJ, coeff_ym1, coeff_yp1, T, J)
    # Plus nonlinear products: 8 terms per constraint
    # Simplest: build one big array of named tuples

    # For efficiency, build product-sum data separately
    # prod_data: for each (T, I, Jx) triple: y=Y(T,I), x=X(T,Jx), c=C(I,Jx)
    # These appear in ALL J constraints for time T. We'll use a structured approach:
    # Add the linear part as separate @add_con and handle nonlinear products.

    # Actually, the cleanest approach: one @add_con with the full expression per (T,J).
    # For each (T,J), we write out the complete formula.

    # Build flattened data:
    con_data = [(
        yTP1J = y_idx(t + 1, j),
        yTJ   = y_idx(t, j),
        yTJm1 = j > 1 ? y_idx(t, j - 1) : 1,
        yTJp1 = j < NY ? y_idx(t, j + 1) : 1,
        xT1   = x_idx(t, 1),
        xT2   = x_idx(t, 2),
        # B coefficients for row J: B(J,1), B(J,2)
        BJ1   = B[j, 1],
        BJ2   = B[j, 2],
        # C matrix for nonlinear: 8 elements = NY*NX
        # y indices for all I: Y(T,1..NY)
        yT1   = y_idx(t, 1),
        yT2   = y_idx(t, 2),
        yT3   = y_idx(t, 3),
        yT4   = y_idx(t, 4),
        # Linear coefficients for Y(T,J), Y(T,J-1), Y(T,J+1)
        cJ    = T(0.5),
        cJm1  = j > 1 ? T(-0.25) : T(0.0),
        cJp1  = j < NY ? T(0.25) : T(0.0),
        # Include Y(T,2)*0.25 for j==1 (special first state)
        # Actually for j==1: Y(T,1)*0.5 + Y(T,2)*0.25
        # The above handles it: cJp1=0.25 when j=1 (j+1=2), yTJp1=y_idx(t,2)
        # For j==NY: Y(T,NY)*0.5 + Y(T,NY-1)*(-0.25) - handled by cJm1=-0.25, cJp1=0
    ) for t in 1:(N-1) for j in 1:NY]

    # C matrix entries for all (I, Jx) pairs at time T, flat for each constraint
    # We'll use @add_par or just embed C values since they're constant
    # Nonlinear sum per constraint: sum_{I=1..4, Jx=1..2} C[I,Jx] * Y(T,I) * X(T,Jx)
    # C is a 4x2 matrix - 8 terms per constraint

    ExaModels.@add_con(c,
        -x[d.yTP1J]
        + d.cJ * x[d.yTJ]
        + d.cJm1 * x[d.yTJm1]
        + d.cJp1 * x[d.yTJp1]
        + d.BJ1 * x[d.xT1]
        + d.BJ2 * x[d.xT2]
        + T(C[1, 1]) * x[d.yT1] * x[d.xT1]
        + T(C[1, 2]) * x[d.yT1] * x[d.xT2]
        + T(C[2, 1]) * x[d.yT2] * x[d.xT1]
        + T(C[2, 2]) * x[d.yT2] * x[d.xT2]
        + T(C[3, 1]) * x[d.yT3] * x[d.xT1]
        + T(C[3, 2]) * x[d.yT3] * x[d.xT2]
        + T(C[4, 1]) * x[d.yT4] * x[d.xT1]
        + T(C[4, 2]) * x[d.yT4] * x[d.xT2]
        for d in con_data)

    return ExaModels.ExaModel(c; kwargs...)
end
