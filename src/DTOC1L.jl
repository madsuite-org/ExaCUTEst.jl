# DTOC1L: Discrete time optimal control, linear transitions (N=1000, NX=2, NY=4)
# Variables: X(T,I) for T=1..N-1, I=1..NX; Y(T,I) for T=1..N, I=1..NY
# Total: (N-1)*NX + N*NY variables, Y(1,I)=0 fixed
# Objective: sum_{T,I} (X(T,I)+0.5)^4 + sum_{T,I} (Y(T,I)+0.25)^4
# Transition constraints TT(T,J) for T=1..N-1, J=1..NY:
#   J=1:   -Y(T+1,1) + 0.5*Y(T,1) + 0.25*Y(T,2) + sum_I B(1,I)*X(T,I) = 0
#   J=2..NY-1: -Y(T+1,J) + 0.5*Y(T,J) - 0.25*Y(T,J-1) + 0.25*Y(T,J+1) + sum B(J,I)*X = 0
#   J=NY: -Y(T+1,NY) + 0.5*Y(T,NY) - 0.25*Y(T,NY-1) + sum B(NY,I)*X = 0
# B(I,J) = (I-J)/(NX+NY)
function dtoc1l_model(N = 1000, NX = 2, NY = 4; T = Float64, backend = nothing, kwargs...)
    NXY = NX + NY
    B = [(i - j) / NXY for i in 1:NY, j in 1:NX]

    NVX = (N - 1) * NX
    NV  = NVX + N * NY

    x_idx(t, i) = (t - 1) * NX + i
    y_idx(t, i) = NVX + (t - 1) * NY + i

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    start = zeros(T, NV)
    for i in 1:NY
        lvar[y_idx(1, i)] = T(0)
        uvar[y_idx(1, i)] = T(0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = start)

    # Objective for X: (X(T,I) + 0.5)^4
    x_obj_data = [(idx = x_idx(t, i),) for t in 1:(N-1) for i in 1:NX]
    ExaModels.@add_obj(c, (x[d.idx] + T(0.5))^4 for d in x_obj_data)
    # Objective for Y: (Y(T,I) + 0.25)^4
    y_obj_data = [(idx = y_idx(t, i),) for t in 1:N for i in 1:NY]
    ExaModels.@add_obj(c, (x[d.idx] + T(0.25))^4 for d in y_obj_data)

    # Constraints for NX=2, NY=4 (default)
    if NX == 2 && NY == 4
        b11 = T(B[1, 1]); b12 = T(B[1, 2])
        b21 = T(B[2, 1]); b22 = T(B[2, 2])
        b31 = T(B[3, 1]); b32 = T(B[3, 2])
        b41 = T(B[4, 1]); b42 = T(B[4, 2])

        t_data = [(yt1_1 = y_idx(t, 1), yt1_2 = y_idx(t, 2),
                   yt1_3 = y_idx(t, 3), yt1_4 = y_idx(t, 4),
                   yt2_1 = y_idx(t + 1, 1), yt2_2 = y_idx(t + 1, 2),
                   yt2_3 = y_idx(t + 1, 3), yt2_4 = y_idx(t + 1, 4),
                   xt_1  = x_idx(t, 1), xt_2  = x_idx(t, 2))
                  for t in 1:(N-1)]

        # TT(T,1): -Y(T+1,1) + 0.5*Y(T,1) + 0.25*Y(T,2) + b11*X(T,1) + b12*X(T,2) = 0
        ExaModels.@add_con(c,
            -x[d.yt2_1] + T(0.5) * x[d.yt1_1] + T(0.25) * x[d.yt1_2] +
            b11 * x[d.xt_1] + b12 * x[d.xt_2] for d in t_data)
        # TT(T,2): -Y(T+1,2) - 0.25*Y(T,1) + 0.5*Y(T,2) + 0.25*Y(T,3) + b21*X(T,1) + b22*X(T,2) = 0
        ExaModels.@add_con(c,
            -x[d.yt2_2] - T(0.25) * x[d.yt1_1] + T(0.5) * x[d.yt1_2] + T(0.25) * x[d.yt1_3] +
            b21 * x[d.xt_1] + b22 * x[d.xt_2] for d in t_data)
        # TT(T,3): -Y(T+1,3) - 0.25*Y(T,2) + 0.5*Y(T,3) + 0.25*Y(T,4) + b31*X(T,1) + b32*X(T,2) = 0
        ExaModels.@add_con(c,
            -x[d.yt2_3] - T(0.25) * x[d.yt1_2] + T(0.5) * x[d.yt1_3] + T(0.25) * x[d.yt1_4] +
            b31 * x[d.xt_1] + b32 * x[d.xt_2] for d in t_data)
        # TT(T,4): -Y(T+1,4) - 0.25*Y(T,3) + 0.5*Y(T,4) + b41*X(T,1) + b42*X(T,2) = 0
        ExaModels.@add_con(c,
            -x[d.yt2_4] - T(0.25) * x[d.yt1_3] + T(0.5) * x[d.yt1_4] +
            b41 * x[d.xt_1] + b42 * x[d.xt_2] for d in t_data)
    else
        error("dtoc1l_model: only NX=2,NY=4 supported")
    end

    return ExaModels.ExaModel(c; kwargs...)
end
