# MANCINO: Mancino's function with variable dimension (default N=100 variables, unconstrained)
# Source: E. Spedicato (1975); Buckley #51; Schittkowski #391 (N=30)
# SIF: N=100 (active), ALPHA=5, BETA=14.0, GAMMA=3
# Variables: X(1)..X(N), all free (FR DEFAULT)
# Groups G(I) with L2 type (no scale): contribution = v_i^2
#   v_i = BETAN * X(I) + sum_{j≠i} MANC(X(J); i, j) - CI
#   where:
#     BETAN = BETA * N = 14.0 * 100 = 1400
#     CI = (i - N/2)^GAMMA = (i - 50)^3
#     MANC(X; i, j) = sqrt(X^2 + i/j) * (sin^5(log(sqrt(X^2+i/j))) + cos^5(log(sqrt(X^2+i/j))))
# Objective: sum_i v_i^2
#
# Implementation: use one @add_par per group (so inner j-index is concrete, avoiding type explosion)
# and one scalar @add_obj per group (100 calls in a Julia loop).
# sum_{j≠i} = sum_{j=1}^N - diagonal_term (since i/i = 1.0 is included in ratios)
function mancino_model(N = 100; T = Float64, backend = nothing, kwargs...)
    ALPHA = 5
    BETA  = T(14.0)
    GAMMA = 3

    # SIF parameter computations
    RN    = T(N)
    BETAN = BETA * RN              # 1400.0

    # Compute scalar A for starting point
    # A = -BETAN / (BETAN^2 - (ALPHA+1)^2*(N-1)^2)
    RN1    = T(N - 1)
    AL1    = T(ALPHA + 1)          # 6.0
    BETAN2 = BETAN * BETAN
    F0     = AL1 * AL1 * RN1 * RN1
    F2     = BETAN2 - F0
    F3     = one(T) / F2
    F4     = BETAN * F3
    A      = -F4

    # Constant -N/2 = -50.0
    HNALF  = RN * T(-0.5)

    # Precompute CI[i] = (i - N/2)^GAMMA = (i - 50)^3
    CI = T[(T(i) + HNALF)^GAMMA for i in 1:N]

    # Compute starting point: X0[i] = A * (H[i] + CI[i])
    # H[i] = sum_{j≠i} sqrt(i/j) * (sin^5(log(sqrt(i/j))) + cos^5(log(sqrt(i/j))))
    x0 = Vector{T}(undef, N)
    for i in 1:N
        RI = T(i)
        H  = zero(T)
        for j in 1:N
            j == i && continue
            RJ    = T(j)
            SQI_J = sqrt(RI / RJ)
            LIJ   = log(SQI_J)
            SIJ   = sin(LIJ)
            CIJ   = cos(LIJ)
            SA    = SIJ^ALPHA
            CA    = CIJ^ALPHA
            H    += SQI_J * (SA + CA)
        end
        x0[i] = A * (H + CI[i])
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # For each group i: add one @add_par for ratios i/j (j=1..N) and one scalar @add_obj
    # By using a separate parameter array per group, the inner j-index is just a concrete
    # integer (not i-dependent), so all SumNode elements share the same expression type.
    # sum_{j≠i} f(x[j]; i/j) = sum_{j=1..N} f(x[j]; ratios[j]) - f(x[i]; 1.0)
    # (ratios[j] = i/j for all j including j=i where it equals 1.0)
    for i in 1:N
        ratios_i = T[T(i) / T(j) for j in 1:N]  # length-N ratios for group i
        ExaModels.@add_par(c, Ri, ratios_i)
        CI_i = CI[i]
        ExaModels.@add_obj(
            c,
            (
                BETAN * x[i]
                + sum(
                    let vij = sqrt(x[j]^2 + Ri[j]);
                        vij * (sin(log(vij))^ALPHA + cos(log(vij))^ALPHA)
                    end
                    for j in 1:N
                )
                - (let vii = sqrt(x[i]^2 + T(1));
                       vii * (sin(log(vii))^ALPHA + cos(log(vii))^ALPHA)
                   end)
                - CI_i
            )^2
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end
