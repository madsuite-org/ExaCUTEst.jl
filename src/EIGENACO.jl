# EIGENACO: Symmetric eigenvalue decomposition A = Q' D Q (constrained optimization version)
# A = diag(1,...,N), N=50 default (SQR2: minimize L2 norm of eigen-equations)
# E(I,J) groups are N-type (objective), O(I,J) groups are E-type (constraints)
# Objective: sum_{I<=J} (sum_K Q(K,I)*D(K)*Q(K,J) - A(I,J))^2
# Constraints: sum_K Q(K,I)*Q(K,J) = delta(I,J), for I<=J
function eigenaco_model(N = 50; T = Float64, backend = nothing, kwargs...)
    D_idx(j)    = j
    Q_idx(i, j) = N + (j - 1) * N + i
    NV = N + N * N

    x0 = zeros(T, NV)
    for j in 1:N
        x0[D_idx(j)]    = T(1.0)
        x0[Q_idx(j, j)] = T(1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # Objective: L2 groups E(I,J) — iterate I<=J (i outer, j inner matches SIF DO I=1..N, DO J=I..N)
    for i in 1:N
        for j in i:N
            aij = T(i == j ? i : 0)
            trip_e = [(qi=Int32(Q_idx(k,i)), dk=Int32(D_idx(k)), qj=Int32(Q_idx(k,j))) for k in 1:N]
            ExaModels.@add_obj(c, (sum(x[d.qi]*x[d.dk]*x[d.qj] for d in trip_e) - aij)^2)
        end
    end

    # Orthogonality constraints O(I,J): sum_K Q(K,I)*Q(K,J) = delta(I,J)
    for i in 1:N
        for j in i:N
            delt = T(i == j ? 1 : 0)
            trip_o = [(qi=Int32(Q_idx(k,i)), qj=Int32(Q_idx(k,j))) for k in 1:N]
            ExaModels.@add_con(c, sum(x[d.qi]*x[d.qj] for d in trip_o) - delt)
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end
