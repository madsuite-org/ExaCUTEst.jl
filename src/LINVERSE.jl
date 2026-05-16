# LINVERSE: Lower bidiagonal matrix inverse approximation, N=1000, 1999 variables, 0 constraints
# Find positive definite lower bidiagonal L s.t. ||L*T*L' - I||_F is minimum
# Variables: A(1)..A(N), B(1)..B(N-1) - 2N-1 variables
# Ordering in SIF: DO I 1 N-1; X A(I); X B(I); ND; X A(N)
# So: x[2i-1] = A(i) for i=1..N-1, x[2i] = B(i) for i=1..N-1, x[2N-1] = A(N)
# A(I) has lower bound EPSILON = 1e-8, all others are free
# Start: all = -1
#
# 2PR element F = X*Y (not 0.5*X*Y, unlike HAIFAS which uses 0.5*X*Y)
#
# Groups O(I,J) with L2 type:
# - Diagonal O(J,J): no SCALE → contribution = v_{J,J}^2
# - Off-diagonal O(J+1,J) and O(J+2,J): SCALE=0.5 → contribution = v^2/0.5 = 2*v^2
#
# The target matrix T(i,j) = sin(i)*cos(j) for |i-j| <= 2, j <= i
# Precompute all T(i,j) values needed
#
# Key formula for groups:
# O(J,J) value = A(J)*A(J)*T(J,J) + A(J)*B(J-1)*T(J,J-1) + B(J-1)*A(J)*T(J,J-1)
#              + B(J-1)*B(J-1)*T(J-1,J-1) - 1 [for J>=2]
# [for J=1] O(1,1) value = A(1)^2 * T(1,1) - 1
#
# This is complex. The key observation: O(I,J) represents the (I,J) entry of L*T*L'.
# Let's just precompute the element contributions and group structure.
#
# For N=1000, we have:
# - 1000 diagonal groups O(J,J) for J=1..N
# - 999 groups O(J+1,J) for J=1..N-1
# - 998 groups O(J+2,J) for J=1..N-2
# Total groups = 2997
#
# Rather than manually encoding each group, we compute the group values algebraically.
# O(J,J) = sum_K (L_{J,K} * (T*L')_{J,K}) - delta_{J,J} = ... (complex)
#
# For lower bidiagonal L: L_{J,J}=A(J), L_{J,J-1}=B(J-1) (zero otherwise)
# (L*T*L')_{i,j} = sum_k sum_l L_{i,k}*T(k,l)*L_{j,l}
# For lower bidiagonal, L_{i,k} nonzero only for k=i or k=i-1
# For row i: contributing terms are k=i (A(i)) and k=i-1 (B(i-1))
# Similarly for column j: l=j (A(j)) and l=j-1 (B(j-1))
# So (L*T*L')_{i,j} = A(i)*A(j)*T(i,j) + A(i)*B(j-1)*T(i,j-1)
#                   + B(i-1)*A(j)*T(i-1,j) + B(i-1)*B(j-1)*T(i-1,j-1)
# Since T is symmetric (T(k,l)=sin(k)cos(l)=T(l,k) only if sin(k)cos(l)=sin(l)cos(k)...
# not in general), but T(i,j) = sin(i)*cos(j) for i>=j only in a pentadiagonal sense.
#
# The group value for O(i,j) = (L*T*L')_{i,j} - delta_{i,j}
# With L2 type (diagonal: contribution = v^2, off-diagonal: 2*v^2)
#
# For efficiency, precompute T values only where needed (pentadiagonal)
function linverse_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    EPSILON = T(1e-8)

    # Variable ordering: A(1), B(1), A(2), B(2), ..., A(N-1), B(N-1), A(N)
    # A(i) → x[2i-1], B(i) → x[2i] for i=1..N-1; A(N) → x[2N-1]
    nvar = 2 * N - 1

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    start = fill(T(-1), nvar)

    # A(I) >= EPSILON: A(i) at position 2i-1 for i=1..N
    for i in 1:N
        lvar[2 * i - 1] = EPSILON
    end

    # Precompute target matrix values T(i,j) = sin(i)*cos(j) for needed indices
    # We need T(i,j) for: |i-j| <= 2, 1<=j<=i<=N
    # Groups O(i,j) for j=i-2..i: need T values up to row/col i, j
    # From SIF: T(J,J), T(J,J-1), T(J+1,J), T(J+2,J), T(J-1,J-1), T(J,J-1)
    # T(i,j) = sin(i)*cos(j) where this is defined for j<=i and |i-j|<=2
    # For the pentadiagonal: T(i,j) defined for j<=i<=j+2

    # Store T values as vectors indexed by (i,j): index = (i-1)*(something)+j
    # We'll use 3 arrays: Tdiag[i]=T(i,i), Toff1[i]=T(i+1,i), Toff2[i]=T(i+2,i)
    Tdiag = T[sin(T(i)) * cos(T(i)) for i in 1:N]
    Toff1 = T[sin(T(i+1)) * cos(T(i)) for i in 1:(N-1)]
    Toff2 = T[sin(T(i+2)) * cos(T(i)) for i in 1:(N-2)]

    # Note: T(i,j) for j<i is also needed as T(j,i)? No: the SIF defines T only for j<=i.
    # But in (L*T*L')_{i,j} we need T(k,l) where k,l can be various pairs.
    # Key: T(i-1,j) where i>j might be needed. Let's just precompute sin and cos vectors.
    sinv = T[sin(T(i)) for i in 1:N]
    cosv = T[cos(T(i)) for i in 1:N]
    # T_val(k,l) = sinv[k]*cosv[l] for l<=k

    # A(i) helper: a_idx(i) = 2i-1; b_idx(i) = 2i (for i=1..N-1)
    a_idx(i) = 2 * i - 1
    b_idx(i) = 2 * i

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    ExaModels.@add_par(c, SIN_V, sinv)
    ExaModels.@add_par(c, COS_V, cosv)

    # Group O(J,J) for J=1..N: diagonal, contribution = v^2, constant rhs=1
    # (L*T*L')_{J,J} = A(J)^2*T(J,J) + 2*A(J)*B(J-1)*T(J,J-1) + B(J-1)^2*T(J-1,J-1) - 1
    # for J=1: no B(0), so = A(1)^2*T(1,1) - 1

    # Diagonal J=1
    ExaModels.@add_obj(c,
        (x[a_idx(1)]^2 * SIN_V[1] * COS_V[1] - T(1))^2)

    # Diagonal J=2..N
    diag_data = [(j = j,
                  td = sinv[j] * cosv[j],          # T(J,J)
                  to = sinv[j] * cosv[j-1],         # T(J,J-1)
                  tp = sinv[j-1] * cosv[j-1]) for j in 2:N]   # T(J-1,J-1)
    ExaModels.@add_par(c, DIAG_TD, T[d.td for d in diag_data])
    ExaModels.@add_par(c, DIAG_TO, T[d.to for d in diag_data])
    ExaModels.@add_par(c, DIAG_TP, T[d.tp for d in diag_data])

    ExaModels.@add_obj(c,
        (x[2 * i + 1]^2 * DIAG_TD[i] + T(2) * x[2 * i + 1] * x[2 * i] * DIAG_TO[i] +
         x[2 * i]^2 * DIAG_TP[i] - T(1))^2
        for i in 1:(N-1))

    # Off-diagonal O(J+1,J) for J=1..N-1: SCALE=0.5, contribution = 2*v^2
    # (L*T*L')_{J+1,J} = A(J+1)*A(J)*T(J+1,J) + A(J+1)*B(J-1)*T(J+1,J-1)
    #                  + B(J)*A(J)*T(J,J) + B(J)*B(J-1)*T(J,J-1)
    # for J=1: no B(0), so = A(2)*A(1)*T(2,1) + B(1)*A(1)*T(1,1)
    ExaModels.@add_obj(c,
        T(2) * (x[a_idx(2)] * x[a_idx(1)] * (sinv[2] * cosv[1]) +
                x[b_idx(1)] * x[a_idx(1)] * (sinv[1] * cosv[1]))^2)

    # J=2..N-2: O(J+1,J) needs B(J-1)
    off1_data = [(j = j,
                  t11 = sinv[j+1] * cosv[j],       # T(J+1,J)
                  t10 = sinv[j+1] * cosv[j-1],      # T(J+1,J-1)
                  t00 = sinv[j] * cosv[j],           # T(J,J)
                  tm0 = sinv[j] * cosv[j-1]) for j in 2:(N-1)]  # T(J,J-1)
    if !isempty(off1_data)
        ExaModels.@add_par(c, OFF1_T11, T[d.t11 for d in off1_data])
        ExaModels.@add_par(c, OFF1_T10, T[d.t10 for d in off1_data])
        ExaModels.@add_par(c, OFF1_T00, T[d.t00 for d in off1_data])
        ExaModels.@add_par(c, OFF1_TM0, T[d.tm0 for d in off1_data])

        ExaModels.@add_obj(c,
            T(2) * (x[2*i+3] * x[2*i+1] * OFF1_T11[i] +
                    x[2*i+3] * x[2*i-2] * OFF1_T10[i] +
                    x[2*i+2] * x[2*i+1] * OFF1_T00[i] +
                    x[2*i+2] * x[2*i-2] * OFF1_TM0[i])^2
            for i in 1:(N-2))
        # Note: for j=i+1 (from off1_data), A(J+1)=x[a_idx(j+1)]=x[2j+1]=x[2*(i+1)+1]=x[2i+3]
        # A(J)=x[a_idx(j)]=x[2j-1]=x[2i+1], B(J)=x[b_idx(j)]=x[2j]=x[2i+2]
        # B(J-1)=x[b_idx(j-1)]=x[2(j-1)]=x[2j-2]=x[2i] -- wait, j starts at 2, so j-1=1, b_idx(1)=2
        # For i=1 (j=2): B(J-1)=B(1)=x[2], so x[2*1+0]=x[2] -- that's x[2*(i-1)+2]? Let me re-check
        # For i=1 (j=2): A(J+1)=A(3)=x[5]=x[2i+3]=x[5]✓; A(J)=A(2)=x[3]=x[2i+1]=x[3]✓
        # B(J)=B(2)=x[4]=x[2i+2]=x[4]✓; B(J-1)=B(1)=x[2]=x[2i-2+2]=x[2(i)+0]?
        # Actually x[2*(i+1)-2] = x[2i]. For i=1: x[2]. Correct. ✓
    end

    # J=N-1: special case (A(J+1)=A(N))
    if N >= 2
        jN1 = N - 1
        t11N = sinv[N] * cosv[jN1]    # T(N,N-1)
        t10N = jN1 >= 2 ? sinv[N] * cosv[jN1-1] : T(0)    # T(N,N-2)
        t00N = sinv[jN1] * cosv[jN1]  # T(N-1,N-1)
        tm0N = jN1 >= 2 ? sinv[jN1] * cosv[jN1-1] : T(0)  # T(N-1,N-2)
        # A(N)=x[2N-1], B(N-1)=x[2(N-1)]=x[2N-2], A(N-1)=x[2(N-1)-1]=x[2N-3]
        # B(N-2)=x[2(N-2)]=x[2N-4] (only exists if N>=3)
        if N >= 3
            ExaModels.@add_obj(c,
                T(2) * (x[2*N-1] * x[2*N-3] * t11N +
                        x[2*N-1] * x[2*N-4] * t10N +
                        x[2*N-2] * x[2*N-3] * t00N +
                        x[2*N-2] * x[2*N-4] * tm0N)^2)
        else  # N=2: no B(N-2)
            ExaModels.@add_obj(c,
                T(2) * (x[2*N-1] * x[2*N-3] * t11N +
                        x[2*N-2] * x[2*N-3] * t00N)^2)
        end
    end

    # Off-diagonal O(J+2,J) for J=1..N-2: SCALE=0.5, contribution = 2*v^2
    # (L*T*L')_{J+2,J} = A(J+2)*A(J)*T(J+2,J) + A(J+2)*B(J-1)*T(J+2,J-1)
    #                  + B(J+1)*A(J)*T(J+1,J) + B(J+1)*B(J-1)*T(J+1,J-1)
    # for J=1: no B(0), so = A(3)*A(1)*T(3,1) + B(2)*A(1)*T(2,1)
    if N >= 3
        ExaModels.@add_obj(c,
            T(2) * (x[a_idx(3)] * x[a_idx(1)] * (sinv[3] * cosv[1]) +
                    x[b_idx(2)] * x[a_idx(1)] * (sinv[2] * cosv[1]))^2)
    end

    # J=2..N-3: O(J+2,J) with B(J-1)
    if N >= 4
        off2_data = [(j = j,
                      t20 = sinv[j+2] * cosv[j],       # T(J+2,J)
                      t2m1 = sinv[j+2] * cosv[j-1],    # T(J+2,J-1)
                      t10 = sinv[j+1] * cosv[j],        # T(J+1,J)
                      t1m1 = sinv[j+1] * cosv[j-1]) for j in 2:(N-3)]
        if !isempty(off2_data)
            ExaModels.@add_par(c, OFF2_T20, T[d.t20 for d in off2_data])
            ExaModels.@add_par(c, OFF2_T2M1, T[d.t2m1 for d in off2_data])
            ExaModels.@add_par(c, OFF2_T10, T[d.t10 for d in off2_data])
            ExaModels.@add_par(c, OFF2_T1M1, T[d.t1m1 for d in off2_data])
            # For j in 2..N-3: A(J+2)=x[a_idx(j+2)], A(J)=x[a_idx(j)], B(J+1)=x[b_idx(j+1)], B(J-1)=x[b_idx(j-1)]
            # Using i=j-1 (i=1..N-4): A(J+2)=x[2(j+2)-1]=x[2j+3]=x[2i+5], A(J)=x[2j-1]=x[2i+1]
            # B(J+1)=x[2(j+1)]=x[2j+2]=x[2i+4], B(J-1)=x[2(j-1)]=x[2j-2]=x[2i]
            ExaModels.@add_obj(c,
                T(2) * (x[2*i+5] * x[2*i+1] * OFF2_T20[i] +
                        x[2*i+5] * x[2*i] * OFF2_T2M1[i] +
                        x[2*i+4] * x[2*i+1] * OFF2_T10[i] +
                        x[2*i+4] * x[2*i] * OFF2_T1M1[i])^2
                for i in 1:(N-4))
        end
    end

    # J=N-2: O(N,N-2), special (A(J+2)=A(N))
    if N >= 3
        jN2 = N - 2
        t20N = sinv[N] * cosv[jN2]         # T(N,N-2)
        t10N = sinv[N-1] * cosv[jN2]       # T(N-1,N-2)
        if jN2 >= 2
            t2m1N = sinv[N] * cosv[jN2-1]      # T(N,N-3)
            t1m1N = sinv[N-1] * cosv[jN2-1]    # T(N-1,N-3)
            # A(N)=x[2N-1], A(N-2)=x[2N-5], B(N-1)=x[2N-2], B(N-3)=x[2N-6]
            ExaModels.@add_obj(c,
                T(2) * (x[2*N-1] * x[2*N-5] * t20N +
                        x[2*N-1] * x[2*N-6] * t2m1N +
                        x[2*N-2] * x[2*N-5] * t10N +
                        x[2*N-2] * x[2*N-6] * t1m1N)^2)
        else  # jN2=1 (N=3): no B(J-1)
            # A(N)=x[5], A(1)=x[1], B(N-1)=x[4]
            ExaModels.@add_obj(c,
                T(2) * (x[2*N-1] * x[1] * t20N +
                        x[2*N-2] * x[1] * t10N)^2)
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end
