# SPINOP: Spin equilibrium with minimum distance objective, N=50 default
# Source: N. Trefethen; SIF input by N. Gould, 2009
# Same as SPIN but with objective: minimize sum_{I>J} V(I,J)^2
#
# CUTEst variable ordering (V variables come first, in reverse declaration order):
#   V(I,J): pos = nV - (I-2)*(I-1)÷2 - J + 1, where nV = N*(N-1)÷2
#   Y(1) = nV+1
#   X(I) = nV+2*(I-1)   for I=2..N
#   Y(I) = nV+2*(I-1)+1 for I=2..N
#   X(1) = nV+2*N, OMEGA = nV+2*N+1, MU = nV+2*N+2
#
# Constraints (E type, SIF order: R, I, M):
#   R(I): -MU*X(I) + OMEGA*Y(I) + sum_{J<I}(Y(I)-Y(J))/V(I,J)^2
#                                - sum_{K>I}(Y(K)-Y(I))/V(K,I)^2 = 0
#   I(I): -MU*Y(I) - OMEGA*X(I) - sum_{J<I}(X(I)-X(J))/V(I,J)^2
#                                + sum_{K>I}(X(K)-X(I))/V(K,I)^2 = 0
#   M(I,J): -V(I,J)^2 + (X(I)-X(J))^2 + (Y(I)-Y(J))^2 = 0
function spinop_model(N = 50; T = Float64, backend = nothing, kwargs...)
    PI     = T(4) * atan(T(1))
    TWO_PI_N = T(2) * PI / T(N)

    nV = N * (N - 1) ÷ 2

    vij(i, j)  = nV - (i - 2) * (i - 1) ÷ 2 - j + 1
    xidx(I)    = I == 1 ? nV + 2*N     : nV + 2*(I-1)
    yidx(I)    = I == 1 ? nV + 1       : nV + 2*(I-1) + 1
    mu_idx     = nV + 2*N + 2
    omega_idx  = nV + 2*N + 1

    nv = nV + 2*N + 2
    startv = fill(T(1), nv)
    for I in 1:N
        startv[xidx(I)] = cos(TWO_PI_N * T(I))
        startv[yidx(I)] = sin(TWO_PI_N * T(I))
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = T(-Inf), uvar = T(Inf), start = startv)

    # Objective: sum_{I>J} V(I,J)^2
    mdata = [(ij = vij(I, J),) for I in 2:N for J in 1:I-1]
    ExaModels.@add_obj(c, x[d.ij]^2 for d in mdata)

    # R(I) constraints — first group in SIF
    for I in 1:N
        xi_I  = xidx(I);  yi_I = yidx(I)
        jlt   = [vij(I, j)  for j in 1:I-1]
        jgt   = [vij(k, I)  for k in I+1:N]
        yj_lt = [yidx(j)    for j in 1:I-1]
        yk_gt = [yidx(k)    for k in I+1:N]
        ExaModels.@add_con(c,
            -x[mu_idx] * x[xi_I] + x[omega_idx] * x[yi_I] +
            sum((x[yi_I] - x[yj_lt[m]]) / x[jlt[m]]^2 for m in 1:length(jlt);
                init = T(0)) -
            sum((x[yk_gt[m]] - x[yi_I]) / x[jgt[m]]^2 for m in 1:length(jgt);
                init = T(0)))
    end
    # I(I) constraints — second group in SIF
    for I in 1:N
        xi_I  = xidx(I);  yi_I = yidx(I)
        jlt   = [vij(I, j)  for j in 1:I-1]
        jgt   = [vij(k, I)  for k in I+1:N]
        xj_lt = [xidx(j)    for j in 1:I-1]
        xk_gt = [xidx(k)    for k in I+1:N]
        ExaModels.@add_con(c,
            -x[mu_idx] * x[yi_I] - x[omega_idx] * x[xi_I] -
            sum((x[xi_I] - x[xj_lt[m]]) / x[jlt[m]]^2 for m in 1:length(jlt);
                init = T(0)) +
            sum((x[xk_gt[m]] - x[xi_I]) / x[jgt[m]]^2 for m in 1:length(jgt);
                init = T(0)))
    end
    # M(I,J) constraints — third group in SIF
    mcon_data = [(I=I, J=J, ij=vij(I,J), xi=xidx(I), yi=yidx(I), xj=xidx(J), yj=yidx(J))
                 for I in 2:N for J in 1:I-1]
    ExaModels.@add_con(c,
        -x[d.ij]^2 + (x[d.xi] - x[d.xj])^2 + (x[d.yi] - x[d.yj])^2
        for d in mcon_data)

    return ExaModels.ExaModel(c; kwargs...)
end
