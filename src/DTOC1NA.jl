# DTOC1NA/NB/NC: Discrete time optimal control, nonlinear transitions
# Same structure as DTOC1L but with nonlinear term:
# C(I,J) = (I+J)/(NX+NY) * MU for I=1..NY, J=1..NX
# Each constraint TT(T,row) gets extra terms: sum_{K=0..NYNX-1} C(I(K),J(K))*X(T,J(K))*Y(T,I(K))
# where K maps to I=K/NX+1, J=K mod NX +1 (column-major, K=0..NYNX-1)
# The element E(T,K) has value C(I,J)*X(T,J)*Y(T,I), and is added to ALL constraints TT(T,row)
# for row=1..NY. That is, each E(T,K) is added to all NY constraints for time step T.
# MU = 0.005 for NA, 0.05 for NB, 0.5 for NC

function _dtoc1n_model(MU; N = 1000, NX = 2, NY = 4, T = Float64, backend = nothing, kwargs...)
    NXY = NX + NY
    B = [(i - j) / NXY for i in 1:NY, j in 1:NX]
    C = [(i + j) * MU / NXY for i in 1:NY, j in 1:NX]

    NVX = (N - 1) * NX
    NV  = NVX + N * NY

    x_idx(t, i) = (t - 1) * NX + i    # control X(T,I)
    y_idx(t, i) = NVX + (t - 1) * NY + i  # state Y(T,I)

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    start = zeros(T, NV)
    for i in 1:NY
        lvar[y_idx(1, i)] = T(0)
        uvar[y_idx(1, i)] = T(0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = start)

    x_obj_data = [(idx = x_idx(t, i),) for t in 1:(N-1) for i in 1:NX]
    ExaModels.@add_obj(c, (x[d.idx] + T(0.5))^4 for d in x_obj_data)
    y_obj_data = [(idx = y_idx(t, i),) for t in 1:N for i in 1:NY]
    ExaModels.@add_obj(c, (x[d.idx] + T(0.25))^4 for d in y_obj_data)

    # Nonlinear element E(T,K): C(I(K),J(K)) * X(T,J(K)) * Y(T,I(K))
    # K=0..NYNX-1: row = K÷NX + 1, col = K mod NX + 1
    # This element is added to ALL constraints TT(T,row) for row=1..NY
    # Actually: GROUP USES loop adds E(T,K) to TT(T,J) for all J and all K
    # i.e., for each T, each constraint TT(T,J) gets sum_{K} E(T,K)
    # For NX=2, NY=4, NYNX=8, K=0..7:
    # K=0: I=1,J=1, K=1: I=1,J=2, K=2: I=2,J=1, ... (column-major: J varies fastest)
    # Wait: I= K/NX + 1, J = K mod NX + 1 (integer div)

    if NX == 2 && NY == 4
        b11 = T(B[1, 1]); b12 = T(B[1, 2])
        b21 = T(B[2, 1]); b22 = T(B[2, 2])
        b31 = T(B[3, 1]); b32 = T(B[3, 2])
        b41 = T(B[4, 1]); b42 = T(B[4, 2])

        # Nonlinear cross terms for each constraint TT(T,J): sum_K C(row_K, col_K)*X(T,col_K)*Y(T,row_K)
        # = C(1,1)*X(T,1)*Y(T,1) + C(1,2)*X(T,1)*Y(T,2)? No wait...
        # K maps as: K=0..NYNX-1, I(K) = K÷NX+1, J(K) = K mod NX+1
        # NYNX=8: K=0: I=1,J=1; K=1: I=1,J=2; K=2: I=2,J=1; K=3: I=2,J=2;
        #          K=4: I=3,J=1; K=5: I=3,J=2; K=6: I=4,J=1; K=7: I=4,J=2
        # E(T,K) is added to TT(T,J) for J=1..NY with XE TT(T,J) E(T,K) for all J
        # Actually looking at GROUP USES: DO T DO J DO K: XE TT(T,J) E(T,K)
        # So each constraint TT(T,J) gets ALL 8 nonlinear elements E(T,0)..E(T,7)
        # The nonlinear sum for each constraint = sum_{K=0..7} C(I_K,J_K)*X(T,J_K)*Y(T,I_K)

        c11 = T(C[1, 1]); c12 = T(C[1, 2])
        c21 = T(C[2, 1]); c22 = T(C[2, 2])
        c31 = T(C[3, 1]); c32 = T(C[3, 2])
        c41 = T(C[4, 1]); c42 = T(C[4, 2])

        # The nonlinear sum (same for ALL rows J!):
        # NL = c11*X1*Y1 + c12*X2*Y1 + c21*X1*Y2 + c22*X2*Y2 +
        #      c31*X1*Y3 + c32*X2*Y3 + c41*X1*Y4 + c42*X2*Y4

        t_data = [(yt1_1 = y_idx(t, 1), yt1_2 = y_idx(t, 2),
                   yt1_3 = y_idx(t, 3), yt1_4 = y_idx(t, 4),
                   yt2_1 = y_idx(t + 1, 1), yt2_2 = y_idx(t + 1, 2),
                   yt2_3 = y_idx(t + 1, 3), yt2_4 = y_idx(t + 1, 4),
                   xt_1  = x_idx(t, 1), xt_2  = x_idx(t, 2))
                  for t in 1:(N-1)]

        # Nonlinear contribution (same for all J):
        # NL(t) = c11*xt1*yt1 + c12*xt2*yt1 + c21*xt1*yt2 + c22*xt2*yt2 +
        #         c31*xt1*yt3 + c32*xt2*yt3 + c41*xt1*yt4 + c42*xt2*yt4

        # TT(T,1): linear + NL = 0
        ExaModels.@add_con(c,
            -x[d.yt2_1] + T(0.5) * x[d.yt1_1] + T(0.25) * x[d.yt1_2] +
            b11 * x[d.xt_1] + b12 * x[d.xt_2] +
            c11 * x[d.xt_1] * x[d.yt1_1] + c12 * x[d.xt_2] * x[d.yt1_1] +
            c21 * x[d.xt_1] * x[d.yt1_2] + c22 * x[d.xt_2] * x[d.yt1_2] +
            c31 * x[d.xt_1] * x[d.yt1_3] + c32 * x[d.xt_2] * x[d.yt1_3] +
            c41 * x[d.xt_1] * x[d.yt1_4] + c42 * x[d.xt_2] * x[d.yt1_4]
            for d in t_data)
        ExaModels.@add_con(c,
            -x[d.yt2_2] - T(0.25) * x[d.yt1_1] + T(0.5) * x[d.yt1_2] + T(0.25) * x[d.yt1_3] +
            b21 * x[d.xt_1] + b22 * x[d.xt_2] +
            c11 * x[d.xt_1] * x[d.yt1_1] + c12 * x[d.xt_2] * x[d.yt1_1] +
            c21 * x[d.xt_1] * x[d.yt1_2] + c22 * x[d.xt_2] * x[d.yt1_2] +
            c31 * x[d.xt_1] * x[d.yt1_3] + c32 * x[d.xt_2] * x[d.yt1_3] +
            c41 * x[d.xt_1] * x[d.yt1_4] + c42 * x[d.xt_2] * x[d.yt1_4]
            for d in t_data)
        ExaModels.@add_con(c,
            -x[d.yt2_3] - T(0.25) * x[d.yt1_2] + T(0.5) * x[d.yt1_3] + T(0.25) * x[d.yt1_4] +
            b31 * x[d.xt_1] + b32 * x[d.xt_2] +
            c11 * x[d.xt_1] * x[d.yt1_1] + c12 * x[d.xt_2] * x[d.yt1_1] +
            c21 * x[d.xt_1] * x[d.yt1_2] + c22 * x[d.xt_2] * x[d.yt1_2] +
            c31 * x[d.xt_1] * x[d.yt1_3] + c32 * x[d.xt_2] * x[d.yt1_3] +
            c41 * x[d.xt_1] * x[d.yt1_4] + c42 * x[d.xt_2] * x[d.yt1_4]
            for d in t_data)
        ExaModels.@add_con(c,
            -x[d.yt2_4] - T(0.25) * x[d.yt1_3] + T(0.5) * x[d.yt1_4] +
            b41 * x[d.xt_1] + b42 * x[d.xt_2] +
            c11 * x[d.xt_1] * x[d.yt1_1] + c12 * x[d.xt_2] * x[d.yt1_1] +
            c21 * x[d.xt_1] * x[d.yt1_2] + c22 * x[d.xt_2] * x[d.yt1_2] +
            c31 * x[d.xt_1] * x[d.yt1_3] + c32 * x[d.xt_2] * x[d.yt1_3] +
            c41 * x[d.xt_1] * x[d.yt1_4] + c42 * x[d.xt_2] * x[d.yt1_4]
            for d in t_data)
    else
        error("_dtoc1n_model: only NX=2, NY=4 supported")
    end

    return ExaModels.ExaModel(c; kwargs...)
end

dtoc1na_model(N = 1000, NX = 2, NY = 4; T = Float64, backend = nothing, kwargs...) =
    _dtoc1n_model(0.005; N = N, NX = NX, NY = NY, T = T, backend = backend, kwargs...)

dtoc1nb_model(N = 1000, NX = 2, NY = 4; T = Float64, backend = nothing, kwargs...) =
    _dtoc1n_model(0.05; N = N, NX = NX, NY = NY, T = T, backend = backend, kwargs...)

dtoc1nc_model(N = 1000, NX = 2, NY = 4; T = Float64, backend = nothing, kwargs...) =
    _dtoc1n_model(0.5; N = N, NX = NX, NY = NY, T = T, backend = backend, kwargs...)
