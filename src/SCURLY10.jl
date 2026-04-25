# SCURLY10: Scaled banded function with semi-bandwidth K=10, N=10000 variables, unconstrained
# Negative curvature near the starting point (scaled version of CURLY10)
# Source: Nick Gould, September 1997.
# SIF: N groups Q(I) with GROUP TYPE P4: F(v) = v*(v*(v^2 - 20) - 0.1)
#   Q(I) = sum_{J=I}^{min(I+K,N)} S(J)*X(J)  where S(J) = exp((J-1)/(N-1)*12)
# All variables free; x0[I] = S(I) + 0.0001*I/(N+1)
# Default N = 10000 (active IE N line), K = 10, SCAL = 12.0
#
# K=10 is fixed: each group Q(i) for i=1..N-K has 11 terms (j=i..i+K).
# Groups i=N-K+1..N are at the tail with fewer terms, expanded individually.
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

    # Groups Q(I) for I=1..N-K: 11 terms (j=i..i+K), GROUP TYPE P4
    # F(v) = v*(v*(v^2 - 20) - 0.1)
    N_K = N - K
    # Use a NamedTuple array to store per-group starting index
    data_bulk = [(i = i,) for i in 1:N_K]
    ExaModels.@add_obj(
        c,
        let v = S_par[d.i]   * x[d.i]    + S_par[d.i+1] * x[d.i+1]  +
                S_par[d.i+2] * x[d.i+2]  + S_par[d.i+3] * x[d.i+3]  +
                S_par[d.i+4] * x[d.i+4]  + S_par[d.i+5] * x[d.i+5]  +
                S_par[d.i+6] * x[d.i+6]  + S_par[d.i+7] * x[d.i+7]  +
                S_par[d.i+8] * x[d.i+8]  + S_par[d.i+9] * x[d.i+9]  +
                S_par[d.i+10] * x[d.i+10]
            v * (v * (v^2 - T(20)) - T(0.1))
        end
        for d in data_bulk
    )

    # Tail groups: I = N-K+1..N, each with fewer terms
    for i in (N_K+1):N
        jend = N  # j goes from i to N
        nterms = jend - i + 1
        # Build the sum expression for nterms terms
        # We pass i as a captured variable (scalar, known at build time)
        S_local = S[i:jend]   # precomputed S values for this group
        data_tail = [(j = j - i + 1,) for j in i:jend]  # relative indices
        # Actually easier: just build one expression with S[i..jend] as parameters
        # Since nterms is small (1..10), we can use a simple @add_par per tail group
        # Simplest: use captured constants since S is known at build time
        c_S = T.(S[i:jend])  # coefficients for x[i..jend]
        # Evaluate sum using @add_obj with closure over c_S
        # ExaModels needs a macro call; easiest is unrolled addition
        # We have at most K=10 terms here; use a loop-based expression
        tail_data = [(ci = c_S[k], xi = i + k - 1) for k in 1:nterms]
        ExaModels.@add_obj(
            c,
            let v = sum(d2.ci * x[d2.xi] for d2 in tail_data)
                v * (v * (v^2 - T(20)) - T(0.1))
            end
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end
