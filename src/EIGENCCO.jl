# EIGENCCO: Symmetric eigenvalue decomposition A = Q' D Q (constrained optimization version)
# Wilkinson's matrix A: N=2M+1 with M=25, N=51 default (SQR2)
# A(I,I) = |M+1-I|, A(I,I+1) = 1 (upper triangle)
# E(I,J) N-type L2 objectives; O(I,J) E-type constraints
function eigencco_model(M = 25; T = Float64, backend = nothing, kwargs...)
    N = 2 * M + 1
    D_idx(j)    = j
    Q_idx(i, j) = N + (j - 1) * N + i
    NV = N + N * N

    A_upper(i, j) = (i == j) ? T(abs(M + 1 - i)) : (i + 1 == j) ? T(1) : T(0)

    x0 = zeros(T, NV)
    for j in 1:N
        x0[D_idx(j)]    = T(1.0)
        x0[Q_idx(j, j)] = T(1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    for i in 1:N
        for j in i:N
            aij = A_upper(i, j)
            trip_e = [(qi=Int32(Q_idx(k,i)), dk=Int32(D_idx(k)), qj=Int32(Q_idx(k,j))) for k in 1:N]
            ExaModels.@add_obj(c, (sum(x[d.qi]*x[d.dk]*x[d.qj] for d in trip_e) - aij)^2)
        end
    end

    for i in 1:N
        for j in i:N
            delt = T(i == j ? 1 : 0)
            trip_o = [(qi=Int32(Q_idx(k,i)), qj=Int32(Q_idx(k,j))) for k in 1:N]
            ExaModels.@add_con(c, sum(x[d.qi]*x[d.qj] for d in trip_o) - delt)
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end
