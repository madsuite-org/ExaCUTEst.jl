# VANDERM1-4: Vandermonde equation systems with monotonicity constraints
# N variables X(1..N), all free, start = (I-1)/N
# N equality constraints E(K) (L2 group type): (sum_I X(I)^K - A(K))^2 = 0
#   E(1): (sum_I X(I) - A(1))^2 = 0  (linear sum)
#   E(K) for K>=2: (sum_I X(I)^K - A(K))^2 = 0  (power sum)
# N-1 monotonicity constraints M(I) for I=2..N: X(I) - X(I-1) >= 0
#
# Variants differ in AL(I) (which determines A(K) = sum_I AL(I)^K):
#   VANDERM1: AL(I) = I/N
#   VANDERM2: AL(I) = 1 + (1-I)/N = (N+1-I)/N
#   VANDERM3: AL(2k-1) = AL(2k) = 2k/N for k=1..(N/2)  [N must be even]
#   VANDERM4: AL(I) alternates 0.5, 2.0, 0.5, 2.0, ...
#
# Source: A. Neumaier, private communication, 1991.
# SIF input: Ph. L. Toint, May 1993.

function vanderm1_model(N = 100; T = Float64, backend = nothing, kwargs...)
    AL = T[i / N for i in 1:N]
    _vanderm_model(N, AL; T = T, backend = backend, kwargs...)
end

function vanderm2_model(N = 100; T = Float64, backend = nothing, kwargs...)
    # AL(I) = 1 + 1/N - I/N = (N+1-I)/N
    AL = T[(N + 1 - i) / N for i in 1:N]
    _vanderm_model(N, AL; T = T, backend = backend, kwargs...)
end

function vanderm3_model(N = 100; T = Float64, backend = nothing, kwargs...)
    # N must be even; AL(2k-1) = AL(2k) = 2k/N for k=1..(N/2)
    @assert iseven(N) "VANDERM3 requires even N"
    AL = Vector{T}(undef, N)
    for k in 1:div(N, 2)
        AL[2k - 1] = T(2k) / T(N)
        AL[2k]     = T(2k) / T(N)
    end
    # A(N) is overridden to 1.0 in the SIF (sum happens to equal 1.0)
    _vanderm_model(N, AL; T = T, backend = backend, kwargs..., vanderm3_override = true)
end

function vanderm4_model(N = 100; T = Float64, backend = nothing, kwargs...)
    # AL alternates: 0.5, 2.0, 0.5, 2.0, ... (TMP starts at 2, then TMP=1/TMP each step)
    AL = Vector{T}(undef, N)
    tmp = T(2)
    for i in 1:N
        tmp = T(1) / tmp
        AL[i] = tmp
    end
    _vanderm_model(N, AL; T = T, backend = backend, kwargs...)
end

function _vanderm_model(N, AL; T = Float64, backend = nothing, vanderm3_override = false, kwargs...)
    # Compute RHS: A(K) = sum_I AL(I)^K
    A = Vector{T}(undef, N)
    for K in 1:N
        A[K] = sum(AL[i]^K for i in 1:N)
    end
    # VANDERM3 overrides A(N) = 1.0
    if vanderm3_override
        A[N] = T(1)
    end

    start_vals = T[(i - 1) / N for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start_vals)

    # Store RHS as parameters
    ExaModels.@add_par(c, A_par, A)

    # E(1): L2 group type → constraint value = (sum_I x[i] - A(1))^2, lcon=ucon=0
    ExaModels.@add_con(c, (sum(x[i] for i in 1:N) - A_par[1])^2)

    # E(K) for K=2..N: (sum_I x[i]^K - A(K))^2 = 0
    # Each is a scalar constraint; add one per K
    for K in 2:N
        rhs_K = A[K]
        ExaModels.@add_con(c, (sum(x[i]^K for i in 1:N) - rhs_K)^2)
    end

    # M(I) for I=2..N: x[I] - x[I-1] >= 0
    mono_data = [(i = i,) for i in 2:N]
    ExaModels.@add_con(c, x[d.i] - x[d.i - 1] for d in mono_data;
                       lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
