# SPIN: Particle spin equilibrium in complex plane, N=50 default
# Source: N. Trefethen; SIF input by N. Gould, 2009
# classification NOR2-AN-V-V (nonlinear system, zero objective)
#
# CUTEst variable ordering (interleaved X/Y after MU, OMEGA):
#   MU=x[1], OMEGA=x[2], X(I)=x[2*I+1], Y(I)=x[2*I+2] for I=1..N
#   V(I,J) = x[vij(I,J)] for I=2..N, J=1..I-1
#   where vij(I,J) = 2 + 2*N + (I-2)*(I-1)÷2 + J
#
# Constraints (E type, SIF order: R, I, M):
#   R(I): -MU*X(I) + OMEGA*Y(I) + sum_{J<I}(Y(I)-Y(J))/V(I,J)^2
#                                - sum_{K>I}(Y(K)-Y(I))/V(K,I)^2 = 0
#   I(I): -MU*Y(I) - OMEGA*X(I) - sum_{J<I}(X(I)-X(J))/V(I,J)^2
#                                + sum_{K>I}(X(K)-X(I))/V(K,I)^2 = 0
#   M(I,J): -V(I,J)^2 + (X(I)-X(J))^2 + (Y(I)-Y(J))^2 = 0
function spin_model(N = 50; T = Float64, backend = nothing, kwargs...)
    PI  = T(4) * atan(T(1))
    TWO_PI_N = T(2) * PI / T(N)

    # V(I,J) index: 2+2N + (I-2)*(I-1)÷2 + J, valid for I>J
    vij = (i, j) -> 2 + 2 * N + (i - 2) * (i - 1) ÷ 2 + j

    nv = 2 + 2 * N + N * (N - 1) ÷ 2
    startv = fill(T(1), nv)
    for I in 1:N
        startv[2*I + 1] = cos(TWO_PI_N * T(I))
        startv[2*I + 2] = sin(TWO_PI_N * T(I))
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = T(-Inf), uvar = T(Inf), start = startv)

    # R(I) constraints — first group in SIF
    for I in 1:N
        jlt = [vij(I, j) for j in 1:I-1]
        jgt = [vij(k, I) for k in I+1:N]
        yj_lt = [2*j + 2 for j in 1:I-1]
        yk_gt = [2*k + 2 for k in I+1:N]
        ExaModels.@add_con(c,
            -x[1] * x[2*I + 1] + x[2] * x[2*I + 2] +
            sum((x[2*I + 2] - x[yj_lt[m]]) / x[jlt[m]]^2 for m in 1:length(jlt);
                init = T(0)) -
            sum((x[yk_gt[m]] - x[2*I + 2]) / x[jgt[m]]^2 for m in 1:length(jgt);
                init = T(0)))
    end
    # I(I) constraints — second group in SIF
    for I in 1:N
        jlt = [vij(I, j) for j in 1:I-1]
        jgt = [vij(k, I) for k in I+1:N]
        xj_lt = [2*j + 1 for j in 1:I-1]
        xk_gt = [2*k + 1 for k in I+1:N]
        ExaModels.@add_con(c,
            -x[1] * x[2*I + 2] - x[2] * x[2*I + 1] -
            sum((x[2*I + 1] - x[xj_lt[m]]) / x[jlt[m]]^2 for m in 1:length(jlt);
                init = T(0)) +
            sum((x[xk_gt[m]] - x[2*I + 1]) / x[jgt[m]]^2 for m in 1:length(jgt);
                init = T(0)))
    end
    # M(I,J) constraints — third group in SIF
    mdata = [(I = I, J = J, ij = vij(I, J)) for I in 2:N for J in 1:I-1]
    ExaModels.@add_con(c,
        -x[d.ij]^2 + (x[2*d.I + 1] - x[2*d.J + 1])^2 + (x[2*d.I + 2] - x[2*d.J + 2])^2
        for d in mdata)

    return ExaModels.ExaModel(c; kwargs...)
end
