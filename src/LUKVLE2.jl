# LUKVLE2: Chained Wood function with Broyden banded constraints
# Source: Problem 5.2, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), 4999 equality constraints
# All variables free; start: odd→-2, even→1
#
# No variable reordering (CUTEst pos = SIF index for all variables).
#
# Objective groups (I=1..N/2-1=4999):
#   A(I): SCALE=0.01, v=-X(2I) + X(2I-1)^2; contrib=100*v^2
#   B(I): v=X(2I-1)-1; contrib=v^2
#   C(I) for I=1..5: SCALE=1/90, v=-X(2I+2)+X(2I+1)^2; contrib=90*v^2
#   C(I) for I=6..4999: SCALE=1/90, v_C includes Broyden banded Broyden terms
#     v_C = -X(2I+2) + X(2I+1)^2
#           + 2*X(I) + sum_{J=I-5}^{I+1} X(J)
#           + 5*X(I)^3 + sum_{J=I-5}^{I+1} X(J)^2 + 1
#     contrib = 90*v_C^2
#   D(I): v=-X(2I+1)-1; contrib=v^2
#   E(I): SCALE=0.1, v=X(2I)+X(2I+2)-2; contrib=10*v^2
#   F(I): SCALE=10.0, v=X(2I)-X(2I-1); contrib=v^2/10
#
# Constraints C(K) for K=5000..9998 (ci=K-4999=1..4999, Broyden banded):
#   2*X(K) + sum_{J=K-5}^{K+1} X(J) + 5*X(K)^3 + sum_{J=K-5}^{K+1} X(J)^2 + 1 = 0
function lukvle2_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert iseven(N) "N must be even"
    N2m1 = N ÷ 2 - 1  # = N/2 - 1

    start = T[isodd(i) ? T(-2) : T(1) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # A(I) and B(I) terms (I=1..N/2-1)
    ExaModels.@add_obj(c, T(100) * (-x[2i] + x[2i-1]^2)^2 for i in 1:N2m1)
    ExaModels.@add_obj(c, (x[2i-1] - T(1))^2 for i in 1:N2m1)

    # D(I) terms: v = -X(2I+1) - 1 (coeff -1, rhs=1 → v=-X-1)
    ExaModels.@add_obj(c, (-x[2i+1] - T(1))^2 for i in 1:N2m1)

    # E(I) terms: SCALE=0.1, contrib=10*v^2
    ExaModels.@add_obj(c, T(10) * (x[2i] + x[2i+2] - T(2))^2 for i in 1:N2m1)

    # F(I) terms: SCALE=10, contrib=v^2/10
    ExaModels.@add_obj(c, (x[2i] - x[2i-1])^2 / T(10) for i in 1:N2m1)

    # C(I=1..5): simple v = -X(2I+2) + X(2I+1)^2
    ExaModels.@add_obj(c, T(90) * (-x[2i+2] + x[2i+1]^2)^2 for i in 1:5)

    # C(I=6..N2m1): Broyden banded formula (I-5 to I+1 band, center at I, Wood cross-terms)
    # v_C = -X(2I+2) + X(2I+1)^2 + 2*X(I) + sum_{J=I-5}^{I+1} X(J) + 5*X(I)^3 + sum_{J=I-5}^{I+1} X(J)^2 + 1
    # For I=6..N2m1: band [I-5, I+1] is fully contained in [1,N] since I-5 >= 1 (I>=6) and I+1 <= N2m1+1 <= N
    if N2m1 >= 7
        ExaModels.@add_obj(
            c,
            T(90) * (
                -x[2i+2] + x[2i+1]^2 +
                T(2) * x[i] + (x[i-5] + x[i-4] + x[i-3] + x[i-2] + x[i-1] + x[i] + x[i+1]) +
                T(5) * x[i]^3 +
                (x[i-5]^2 + x[i-4]^2 + x[i-3]^2 + x[i-2]^2 + x[i-1]^2 + x[i]^2 + x[i+1]^2) +
                T(1)
            )^2 for i in 6:N2m1
        )
    end

    # Constraints C(K) for K=5000..N-2 (Broyden banded, band [K-5,K+1]):
    # ci = K - 4999; K = ci + 4999
    # band: [K-5, K+1] = [ci+4994, ci+5000], center=ci+4999
    # For all K=5000..9998: K-5=4995..9993 >= 1 and K+1=5001..9999 <= N ✓
    ExaModels.@add_con(
        c,
        T(2) * x[k+4999] +
        (x[k+4994] + x[k+4995] + x[k+4996] + x[k+4997] + x[k+4998] + x[k+4999] + x[k+5000]) +
        T(5) * x[k+4999]^3 +
        (x[k+4994]^2 + x[k+4995]^2 + x[k+4996]^2 + x[k+4997]^2 + x[k+4998]^2 + x[k+4999]^2 + x[k+5000]^2) +
        T(1)
        for k in 1:(N2m1 - 1)
    )

    return ExaModels.ExaModel(c; kwargs...)
end
