# GAUSSELM: Maximum pivot growth in Gaussian elimination, N=40 default
# Variables X(I,J,K) for K=1..N, J=K..N, I=K..N
# Minimize -X(N,N,N)
# Constraints: X(I,J,K+1) = X(I,J,K) - X(I,K,K)*X(K,J,K)/X(K,K,K) for K=1..N-1, I,J=K+1..N
# Bounds: |X(I,J,1)| <= 1, X(K,K,K) >= 0, X(1,1,1) = 1 (fixed)
function gausselm_model(N = 40; T = Float64, backend = nothing, kwargs...)
    # Count total variables
    NV = sum((N - k + 1)^2 for k in 1:N)

    # Build a mapping from (I,J,K) to flat index
    # Variables declared as: DO K=1..N, DO J=K..N, DO I=K..N: X(I,J,K)
    # So order is: for K=1: all (I,J) with I,J in K..N; for K=2: all (I,J) with I,J in K..N; etc.
    idx_map = Dict{Tuple{Int,Int,Int}, Int}()
    cnt = 0
    for k in 1:N
        for j in k:N
            for i in k:N
                cnt += 1
                idx_map[(i, j, k)] = cnt
            end
        end
    end

    # Starting point: X(I,J,K) = 0.95*(I+J+K)/(6N) for all variables
    x0 = zeros(T, NV)
    for (ijk, idx) in idx_map
        i, j, k = ijk
        x0[idx] = T(0.95) * T(i + j + k) / T(6 * N)
    end

    # Bounds
    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)

    # |X(I,J,1)| <= 1 for all I,J in 1..N
    for j in 1:N
        for i in 1:N
            idx = idx_map[(i, j, 1)]
            lvar[idx] = T(-1)
            uvar[idx] = T(1)
        end
    end

    # X(K,K,K) >= 0 for K=1..N
    for k in 1:N
        lvar[idx_map[(k, k, k)]] = T(0)
    end

    # X(1,1,1) = 1 (fixed)
    lvar[idx_map[(1, 1, 1)]] = T(1)
    uvar[idx_map[(1, 1, 1)]] = T(1)
    x0[idx_map[(1, 1, 1)]] = T(1)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: minimize -X(N,N,N)
    ExaModels.@add_obj(c, -x[idx_map[(N, N, N)]])

    # Constraints E(I,J,K) for K=1..N-1, I=K+1..N, J=K+1..N:
    # X(I,J,K+1) - X(I,J,K) + X(I,K,K)*X(K,J,K)/X(K,K,K) = 0
    e_data = [(
        xijk1 = idx_map[(i, j, k + 1)],
        xijk  = idx_map[(i, j, k)],
        xikk  = idx_map[(i, k, k)],
        xkjk  = idx_map[(k, j, k)],
        xkkk  = idx_map[(k, k, k)],
    ) for k in 1:(N - 1) for i in (k + 1):N for j in (k + 1):N]

    ExaModels.@add_con(c,
        x[d.xijk1] - x[d.xijk] + x[d.xikk] * x[d.xkjk] / x[d.xkkk]
        for d in e_data)

    # Complete pivoting constraints |X(I,J,K)| <= X(K,K,K) for K=2..N-1
    # M(I,K,K): X(I,K,K) - X(K,K,K) <= 0 (L-type)
    # P(I,K,K): X(I,K,K) + X(K,K,K) >= 0 (G-type)
    # M(K,I,K): X(K,I,K) - X(K,K,K) <= 0
    # P(K,I,K): X(K,I,K) + X(K,K,K) >= 0
    # M(I,J,K): X(I,J,K) - X(K,K,K) <= 0 for J=K+1..N
    # P(I,J,K): X(I,J,K) + X(K,K,K) >= 0

    piv_data = [(
        xijk = idx_map[(i, j, k)],
        xkkk = idx_map[(k, k, k)],
    ) for k in 2:(N - 1) for i in (k + 1):N for j in (k + 1):N]

    ExaModels.@add_con(c,
        x[d.xijk] - x[d.xkkk]
        for d in piv_data;
        lcon = T(-Inf), ucon = T(0))

    ExaModels.@add_con(c,
        x[d.xijk] + x[d.xkkk]
        for d in piv_data;
        lcon = T(0), ucon = T(Inf))

    # Column pivoting: M(I,K,K) and P(I,K,K) for K=2..N-1, I=K+1..N
    col_piv_data = [(
        xikk = idx_map[(i, k, k)],
        xkkk = idx_map[(k, k, k)],
    ) for k in 2:(N - 1) for i in (k + 1):N]

    ExaModels.@add_con(c,
        x[d.xikk] - x[d.xkkk]
        for d in col_piv_data;
        lcon = T(-Inf), ucon = T(0))

    ExaModels.@add_con(c,
        x[d.xikk] + x[d.xkkk]
        for d in col_piv_data;
        lcon = T(0), ucon = T(Inf))

    # Row pivoting: M(K,I,K) and P(K,I,K) for K=2..N-1, I=K+1..N
    row_piv_data = [(
        xkik = idx_map[(k, i, k)],
        xkkk = idx_map[(k, k, k)],
    ) for k in 2:(N - 1) for i in (k + 1):N]

    ExaModels.@add_con(c,
        x[d.xkik] - x[d.xkkk]
        for d in row_piv_data;
        lcon = T(-Inf), ucon = T(0))

    ExaModels.@add_con(c,
        x[d.xkik] + x[d.xkkk]
        for d in row_piv_data;
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
