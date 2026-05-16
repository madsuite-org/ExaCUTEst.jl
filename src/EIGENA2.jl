# EIGENA2: Symmetric eigenvalue decomposition (constrained optimization version 2)
# A = diag(1,...,N), N=50 default (QQR2)
# Variables: D(J) eigenvalues (J=1..N) + Q(I,J) orthogonal matrix (I,J=1..N)
# Variable layout: D(J) at index J; Q(I,J) at index N + (J-1)*N + I
#
# GROUPS section (SIF order):
#   DO J=1..N, DO I=1..J:
#     O(I,J) constraint (XE): sum_K Q(K,I)*Q(K,J) = delta(I,J)
#   DO J=1..N, DO I=1..N:
#     E(I,J) objective (ZN, L2 type): element=Q(J,I)*D(J), linear=-I*Q(J,I)
#     L2 contribution = (Q(J,I)*D(J) - I*Q(J,I))^2 = (Q(J,I)*(D(J)-I))^2
#
# Note: for diagonal A, A(I,K)=I if K=I else 0, so linear part = -I*Q(J,I)
function eigena2_model(N = 50; T = Float64, backend = nothing, kwargs...)
    D_idx(j)    = j
    Q_idx(i, j) = N + (j - 1) * N + i  # Q(I,J): row I, column J
    NV = N + N * N

    x0 = zeros(T, NV)
    for j in 1:N
        x0[D_idx(j)]    = T(1.0)
        x0[Q_idx(j, j)] = T(1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # O(I,J) constraints: sum_K Q(K,I)*Q(K,J) - delta(I,J) = 0
    # Loop order matches SIF GROUPS: DO J=1..N, DO I=1..J
    for j in 1:N
        for i in 1:j
            delt = T(i == j ? 1 : 0)
            trip_o = [(qi=Int32(Q_idx(k,i)), qj=Int32(Q_idx(k,j))) for k in 1:N]
            ExaModels.@add_con(c, sum(x[d.qi]*x[d.qj] for d in trip_o) - delt)
        end
    end

    # E(I,J) objectives: L2 type, for J=1..N, I=1..N
    # Value = Q(J,I)*D(J) - I*Q(J,I) = Q(J,I)*(D(J) - I)
    # L2 contribution = (Q(J,I)*(D(J) - I))^2
    # Q(J,I) in our layout = Q_idx(J, I) = N + (I-1)*N + J
    e_data = [(
        qji = Int32(N + (i - 1) * N + j),  # Q(J,I) = Q_idx(J, I)
        dj  = Int32(D_idx(j)),
        ai  = T(i),  # A(I,I) = I for diagonal matrix
    ) for j in 1:N for i in 1:N]
    ExaModels.@add_obj(c,
        (x[d.qji] * x[d.dj] - d.ai * x[d.qji])^2
        for d in e_data)

    return ExaModels.ExaModel(c; kwargs...)
end
