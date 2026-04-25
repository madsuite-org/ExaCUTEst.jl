# SCURLY10: Scaled banded function with semi-bandwidth K=10, N=10000 variables, unconstrained
# Negative curvature near the starting point (scaled version of CURLY10)
# Source: Nick Gould, September 1997.
# SIF: N groups Q(I) with GROUP TYPE P4: F(v) = v*(v*(v^2 - 20) - 0.1)
#   Q(I) = sum_{J=I}^{min(I+K,N)} S(J)*X(J)  where S(J) = exp((J-1)/(N-1)*12)
# All variables free; x0[I] = S(I) + 0.0001*I/(N+1)
# Default N = 10000 (active IE N line), K = 10, SCAL = 12.0
function scurly10_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    K = 10
    SCAL = T(12.0)
    RN = T(N)
    RN_1 = RN - T(1)   # N-1
    RN_p1 = RN + T(1)  # N+1

    # Scale factors: S[I] = exp((I-1)/(N-1)*SCAL), I=1..N
    S = T[exp(T(I - 1) / RN_1 * SCAL) for I in 1:N]

    # Start point: x0[I] = S[I] + 0.0001*I/(N+1)
    x0 = T[S[I] + T(0.0001) * T(I) / RN_p1 for I in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)
    ExaModels.@add_par(c, S_par, S)

    # Groups Q(I) for I=1..N-K: sum_{J=I}^{I+K} S[J]*X[J]  (K+1 = 11 terms)
    # Groups Q(I) for I=N-K+1..N: sum_{J=I}^{N} S[J]*X[J]  (fewer terms)
    # GROUP TYPE P4: F(v) = v*(v*(v^2-20)-0.1) = v^4 - 20*v^2 - 0.1*v
    N_K = N - K        # N - K
    ExaModels.@add_obj(
        c,
        let v = sum(S_par[j] * x[j] for j in i:i+K)
            v * (v * (v^2 - T(20)) - T(0.1))
        end
        for i in 1:N_K
    )
    # Tail groups: I = N-K+1..N
    for i in (N_K+1):N
        jend = N
        ExaModels.@add_obj(
            c,
            let v = sum(S_par[j] * x[j] for j in i:jend)
                v * (v * (v^2 - T(20)) - T(0.1))
            end
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end
