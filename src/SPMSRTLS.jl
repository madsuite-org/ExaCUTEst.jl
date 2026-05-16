# SPMSRTLS: Tridiagonal matrix square root problem (Liu & Nocedal / Buckley #151)
# Least-squares (unconstrained) variant of SPMSQRT.
# M=1667 (default), n = 3M-2 = 4999 variables X(I,J) with |I-J| <= 1
#
# B is tridiagonal: B(I,J) for |I-J|<=1, filled sequentially with sin(k^2)
#   k=1: B(1,1)=sin(1); k=2: B(1,2)=sin(4); k=3: B(2,1)=sin(9); ...
# Target: A = B*B (pentadiagonal)
# Objective: sum_{(I,J) in pentadiagonal pattern} (sum_{K valid} X(I,K)*X(K,J) - A(I,J))^2
# Group type L2. All variables free (FR DEFAULT). Starting point: 0.2*B.
function spmsrtls_model(M = 1667; T = Float64, backend = nothing, kwargs...)
    NVAR = 3 * M - 2

    # Variable index for X(I,J) in the tridiagonal, |I-J|<=1
    function xidx(i, j)
        if i == 1
            j == 1 ? 1 : 2
        elseif i == M
            j == i - 1 ? 3 * M - 3 : 3 * M - 2
        else
            j == i - 1 ? 3 * i - 3 : (j == i ? 3 * i - 2 : 3 * i - 1)
        end
    end

    # Build tridiagonal B matrix using sin(k^2) sequentially
    B = zeros(T, M, M)
    kk = 1
    B[1, 1] = sin(T(kk)^2); kk += 1
    B[1, 2] = sin(T(kk)^2); kk += 1
    for i in 2:M-1
        B[i, i-1] = sin(T(kk)^2); kk += 1
        B[i, i]   = sin(T(kk)^2); kk += 1
        B[i, i+1] = sin(T(kk)^2); kk += 1
    end
    B[M, M-1] = sin(T(kk)^2); kk += 1
    B[M, M]   = sin(T(kk)^2)

    # Compute A(I,J) = sum_K B[I,K]*B[K,J] for the valid K range
    function a_val(ii, jj)
        klo = max(1, max(ii - 1, jj - 1))
        khi = min(M, min(ii + 1, jj + 1))
        return sum(B[ii, kk] * B[kk, jj] for kk in klo:khi)
    end

    # Groups enumeration: pentadiagonal entries that appear in GROUP USES
    # Row 1: J=1,2,3
    # Row 2: J=1,2,3,4
    # Row I (3..M-2): J=I-2..I+2
    # Row M-1: J=M-3,M-2,M-1,M
    # Row M:   J=M-2,M-1,M
    #
    # For each group, number of valid K terms:
    # Corner entries (e.g., (1,3), (1,1)): 1 term
    # Edge entries: 2 terms
    # Interior entries: 3 terms
    # Separate by term count for fixed ExaModels kernel structure.

    p_1  = Int[]; q_1  = Int[]; a_1  = T[]  # 1-term groups
    p1_2 = Int[]; q1_2 = Int[]
    p2_2 = Int[]; q2_2 = Int[]; a_2  = T[]  # 2-term groups
    p1_3 = Int[]; q1_3 = Int[]
    p2_3 = Int[]; q2_3 = Int[]
    p3_3 = Int[]; q3_3 = Int[]; a_3  = T[]  # 3-term groups

    function add_group!(ii, jj)
        klo = max(1, max(ii - 1, jj - 1))
        khi = min(M, min(ii + 1, jj + 1))
        ks = klo:khi
        aval = a_val(ii, jj)
        nk = length(ks)
        if nk == 1
            push!(p_1, xidx(ii, ks[1])); push!(q_1, xidx(ks[1], jj)); push!(a_1, aval)
        elseif nk == 2
            push!(p1_2, xidx(ii, ks[1])); push!(q1_2, xidx(ks[1], jj))
            push!(p2_2, xidx(ii, ks[2])); push!(q2_2, xidx(ks[2], jj))
            push!(a_2, aval)
        else
            push!(p1_3, xidx(ii, ks[1])); push!(q1_3, xidx(ks[1], jj))
            push!(p2_3, xidx(ii, ks[2])); push!(q2_3, xidx(ks[2], jj))
            push!(p3_3, xidx(ii, ks[3])); push!(q3_3, xidx(ks[3], jj))
            push!(a_3, aval)
        end
    end

    for jj in 1:3;         add_group!(1, jj); end
    for jj in 1:4;         add_group!(2, jj); end
    for ii in 3:M-2, jj in ii-2:ii+2; add_group!(ii, jj); end
    for jj in M-3:M;       add_group!(M-1, jj); end
    for jj in M-2:M;       add_group!(M, jj); end

    # Starting point: 0.2 * B values (in variable order)
    x0 = zeros(T, NVAR)
    x0[1] = T(0.2) * B[1, 1]
    x0[2] = T(0.2) * B[1, 2]
    for i in 2:M-1
        x0[xidx(i, i-1)] = T(0.2) * B[i, i-1]
        x0[xidx(i, i)]   = T(0.2) * B[i, i]
        x0[xidx(i, i+1)] = T(0.2) * B[i, i+1]
    end
    x0[xidx(M, M-1)] = T(0.2) * B[M, M-1]
    x0[xidx(M, M)]   = T(0.2) * B[M, M]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NVAR; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # 1-term groups: (x[p]*x[q] - a)^2
    if !isempty(p_1)
        data1 = [(p=Int32(p_1[i]), q=Int32(q_1[i]), a=a_1[i]) for i in 1:length(p_1)]
        ExaModels.@add_obj(c, (x[d.p] * x[d.q] - d.a)^2 for d in data1)
    end

    # 2-term groups: (x[p1]*x[q1] + x[p2]*x[q2] - a)^2
    if !isempty(p1_2)
        data2 = [(p1=Int32(p1_2[i]), q1=Int32(q1_2[i]), p2=Int32(p2_2[i]), q2=Int32(q2_2[i]), a=a_2[i]) for i in 1:length(p1_2)]
        ExaModels.@add_obj(c,
            (x[d.p1] * x[d.q1] + x[d.p2] * x[d.q2] - d.a)^2
            for d in data2)
    end

    # 3-term groups: (x[p1]*x[q1] + x[p2]*x[q2] + x[p3]*x[q3] - a)^2
    if !isempty(p1_3)
        data3 = [(p1=Int32(p1_3[i]), q1=Int32(q1_3[i]), p2=Int32(p2_3[i]), q2=Int32(q2_3[i]), p3=Int32(p3_3[i]), q3=Int32(q3_3[i]), a=a_3[i]) for i in 1:length(p1_3)]
        ExaModels.@add_obj(c,
            (x[d.p1] * x[d.q1] + x[d.p2] * x[d.q2] + x[d.p3] * x[d.q3] - d.a)^2
            for d in data3)
    end

    return ExaModels.ExaModel(c; kwargs...)
end
