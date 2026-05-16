# DTOC2: Discrete-time optimal control problem 2
# N=1000 periods, NX=2 controls, NY=4 states; all variables free
# Objective O(T) for T=1..N uses OEL element (nonlinear in X,Y) + O(N) uses YNSQ elements
# Constraints TT(T,J) = -Y(T+1,J) + SINE(Y(T,J)) + sum_I C(J,I)*SINE(X(T,I)) = 0
# OEL(T): (Y1^2+Y2^2+Y3^2+Y4^2)*(sin^2(0.5*(X1^2+X2^2))+1) = YN2 * SZ2
# SIF: C(I,J)=(I+J)/(2*NY) for I=1..NY, J=1..NX
function dtoc2_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    NX = 2
    NY = 4
    NY2 = NY * 2  # 8
    RNY2 = T(1.0 / NY2)

    # C(I,J) = (I+J)*1/(2*NY) for I=1..NY, J=1..NX
    C = [T(I + J) * RNY2 for I in 1:NY, J in 1:NX]

    # Y(1,I) start point: i/(2*NY) for I=1..NY (also fixed? No, only start point)
    # bounds: all free

    # Variable layout: X(T,I) T=1..N-1, I=1..NX; Y(T,I) T=1..N, I=1..NY
    NX_total = (N - 1) * NX
    NV = NX_total + N * NY

    x_idx(t, i) = (t - 1) * NX + i
    y_idx(t, i) = NX_total + (t - 1) * NY + i

    x0 = zeros(T, NV)
    for i in 1:NY
        x0[y_idx(1, i)] = T(i) * RNY2
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # Objective O(T) for T=1..N-1: OEL element
    # OEL = (Y1^2+Y2^2+Y3^2+Y4^2) * (sin^2(0.5*(X1^2+X2^2)) + 1)
    odata = [(
        y1 = y_idx(t, 1), y2 = y_idx(t, 2), y3 = y_idx(t, 3), y4 = y_idx(t, 4),
        x1 = x_idx(t, 1), x2 = x_idx(t, 2),
    ) for t in 1:(N-1)]
    ExaModels.@add_obj(c,
        (x[d.y1]^2 + x[d.y2]^2 + x[d.y3]^2 + x[d.y4]^2) *
        (sin(T(0.5) * (x[d.x1]^2 + x[d.x2]^2))^2 + T(1))
        for d in odata)

    # Objective O(N): sum_J Y(N,J)^2
    yn_data = [(yi = y_idx(N, i),) for i in 1:NY]
    ExaModels.@add_obj(c, x[d.yi]^2 for d in yn_data)

    # Constraints TT(T,J) for T=1..N-1, J=1..NY
    # -Y(T+1,J) + sin(Y(T,J)) + sum_I C(J,I)*sin(X(T,I)) = 0
    cdata = [(
        yTP1J = y_idx(t + 1, j),
        yTJ   = y_idx(t, j),
        xT1   = x_idx(t, 1),
        xT2   = x_idx(t, 2),
        CJ1   = C[j, 1],
        CJ2   = C[j, 2],
    ) for t in 1:(N-1) for j in 1:NY]
    ExaModels.@add_con(c,
        -x[d.yTP1J] + sin(x[d.yTJ])
        + d.CJ1 * sin(x[d.xT1])
        + d.CJ2 * sin(x[d.xT2])
        for d in cdata)

    return ExaModels.ExaModel(c; kwargs...)
end
