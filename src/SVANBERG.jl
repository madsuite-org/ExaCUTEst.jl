# SVANBERG: Structural optimization subproblem (Method of Moving Asymptotes)
# N variables (must be even, >= 10), N equality/inequality constraints
# Default N = 10 (for testing; SIF default is 5000)
# Variables: x[1..N], all bounded to [-0.8, 0.8]
# Objective: sum_{k=1}^{N/2} [A(2k-1)/(1+x[2k-1]) + A(2k)/(1-x[2k])]
#   where A(odd I) = 2*I/N + 1, A(even I) = -3*I/N + 5
# Constraints C(I) <= B(I), where B(I) = 5*I/N + 10, each involving 9 specific
# EP/EM element values from different indices.
# EP(I) = 1/(1+x[I]), EM(I) = 1/(1-x[I])
# SIF: OOR2-MN-V-V. Source: Svanberg K. (1987), Int.J. Num. Meth. Eng. 24.
function svanberg_model(N = 10; T = Float64, backend = nothing, kwargs...)
    @assert N >= 10 && iseven(N) "N must be even and >= 10"

    RN = T(N)
    B = T[T(i) * T(5) / RN + T(10) for i in 1:N]  # B(I) = I*5/N + 10

    # Objective coefficients A(I):
    # For odd I: A(I) = 2*I/N + 1
    # For even I: A(I) = -3*I/N + 5
    A_odd  = T[T(2 * i) / RN + T(1) for i in 1:2:N]  # A for odd indices 1,3,...,N-1
    A_even = T[-T(3 * i) / RN + T(5) for i in 2:2:N]  # A for even indices 2,4,...,N

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    ExaModels.@add_var(c, x, N;
        lvar  = fill(T(-0.8), N),
        uvar  = fill(T(0.8), N),
        start = fill(T(0), N))

    # Objective: sum over pairs k=1..N/2
    ExaModels.@add_par(c, Aodd_par, A_odd)
    ExaModels.@add_par(c, Aeven_par, A_even)
    ExaModels.@add_obj(c, Aodd_par[k] / (1 + x[2*k - 1]) + Aeven_par[k] / (1 - x[2*k])
                       for k in 1:(N÷2))

    # Corner constraints (hardcoded for general N >= 10):
    # C(1): P(1)+P(2)+Q(3)+P(4)+Q(5)+Q(N-3)+P(N-2)+P(N-1)+Q(N) ≤ B(1)
    ExaModels.@add_con(c,
        1/(1-x[1]) + 1/(1-x[2]) + 1/(1+x[3]) + 1/(1-x[4]) + 1/(1+x[5]) +
        1/(1+x[N-3]) + 1/(1-x[N-2]) + 1/(1-x[N-1]) + 1/(1+x[N]) - B[1];
        lcon = T(-Inf), ucon = T(0))

    # C(2): P(1)+Q(2)+Q(3)+P(4)+Q(5)+P(6)+P(N-2)+Q(N-1)+Q(N) ≤ B(2)
    ExaModels.@add_con(c,
        1/(1-x[1]) + 1/(1+x[2]) + 1/(1+x[3]) + 1/(1-x[4]) + 1/(1+x[5]) + 1/(1-x[6]) +
        1/(1-x[N-2]) + 1/(1+x[N-1]) + 1/(1+x[N]) - B[2];
        lcon = T(-Inf), ucon = T(0))

    # C(3): P(1)+Q(2)+P(3)+P(4)+Q(5)+P(6)+Q(7)+Q(N-1)+P(N) ≤ B(3)
    ExaModels.@add_con(c,
        1/(1-x[1]) + 1/(1+x[2]) + 1/(1-x[3]) + 1/(1-x[4]) + 1/(1+x[5]) + 1/(1-x[6]) +
        1/(1+x[7]) + 1/(1+x[N-1]) + 1/(1-x[N]) - B[3];
        lcon = T(-Inf), ucon = T(0))

    # C(4): Q(1)+Q(2)+P(3)+Q(4)+Q(5)+P(6)+Q(7)+P(8)+P(N) ≤ B(4)
    ExaModels.@add_con(c,
        1/(1+x[1]) + 1/(1+x[2]) + 1/(1-x[3]) + 1/(1+x[4]) + 1/(1+x[5]) + 1/(1-x[6]) +
        1/(1+x[7]) + 1/(1-x[8]) + 1/(1-x[N]) - B[4];
        lcon = T(-Inf), ucon = T(0))

    # Loop: C(I) and C(I+1) for I=5,7,...,N-5 step 2
    # Emit interleaved pairs in order: C(5),C(6), C(7),C(8), ..., C(N-5),C(N-4)
    #   C(I): Q(I-4)+P(I-3)+P(I-2)+Q(I-1)+P(I)+P(I+1)+Q(I+2)+P(I+3)+Q(I+4) ≤ B(I)
    #   C(I+1): P(I-3)+Q(I-2)+Q(I-1)+P(I)+Q(I+1)+Q(I+2)+P(I+3)+Q(I+4)+P(I+5) ≤ B(I+1)
    for i in 5:2:(N-5)
        ExaModels.@add_con(c,
            1/(1+x[i-4]) + 1/(1-x[i-3]) + 1/(1-x[i-2]) + 1/(1+x[i-1]) +
            1/(1-x[i]) + 1/(1-x[i+1]) + 1/(1+x[i+2]) + 1/(1-x[i+3]) +
            1/(1+x[i+4]) - B[i];
            lcon = T(-Inf), ucon = T(0))
        ExaModels.@add_con(c,
            1/(1-x[i-3]) + 1/(1+x[i-2]) + 1/(1+x[i-1]) + 1/(1-x[i]) +
            1/(1+x[i+1]) + 1/(1+x[i+2]) + 1/(1-x[i+3]) + 1/(1+x[i+4]) +
            1/(1-x[i+5]) - B[i+1];
            lcon = T(-Inf), ucon = T(0))
    end

    # Corner constraints C(N-3) through C(N):
    # C(N-3): Q(1)+Q(N-7)+P(N-6)+P(N-5)+Q(N-4)+P(N-3)+P(N-2)+Q(N-1)+P(N) ≤ B(N-3)
    ExaModels.@add_con(c,
        1/(1+x[1]) + 1/(1+x[N-7]) + 1/(1-x[N-6]) + 1/(1-x[N-5]) + 1/(1+x[N-4]) +
        1/(1-x[N-3]) + 1/(1-x[N-2]) + 1/(1+x[N-1]) + 1/(1-x[N]) - B[N-3];
        lcon = T(-Inf), ucon = T(0))

    # C(N-2): Q(1)+P(2)+P(N-6)+Q(N-5)+Q(N-4)+P(N-3)+Q(N-2)+Q(N-1)+P(N) ≤ B(N-2)
    ExaModels.@add_con(c,
        1/(1+x[1]) + 1/(1-x[2]) + 1/(1-x[N-6]) + 1/(1+x[N-5]) + 1/(1+x[N-4]) +
        1/(1-x[N-3]) + 1/(1+x[N-2]) + 1/(1+x[N-1]) + 1/(1-x[N]) - B[N-2];
        lcon = T(-Inf), ucon = T(0))

    # C(N-1): Q(1)+P(2)+Q(3)+Q(N-5)+P(N-4)+P(N-3)+Q(N-2)+P(N-1)+P(N) ≤ B(N-1)
    ExaModels.@add_con(c,
        1/(1+x[1]) + 1/(1-x[2]) + 1/(1+x[3]) + 1/(1+x[N-5]) + 1/(1-x[N-4]) +
        1/(1-x[N-3]) + 1/(1+x[N-2]) + 1/(1-x[N-1]) + 1/(1-x[N]) - B[N-1];
        lcon = T(-Inf), ucon = T(0))

    # C(N): Q(1)+P(2)+Q(3)+P(4)+P(N-4)+Q(N-3)+Q(N-2)+P(N-1)+Q(N) ≤ B(N)
    ExaModels.@add_con(c,
        1/(1+x[1]) + 1/(1-x[2]) + 1/(1+x[3]) + 1/(1-x[4]) + 1/(1-x[N-4]) +
        1/(1+x[N-3]) + 1/(1+x[N-2]) + 1/(1-x[N-1]) + 1/(1+x[N]) - B[N];
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
