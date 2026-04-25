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
# Encoding: sum_{j≠i} = sum_{j=1}^{N} - (j=i term), since i/i=1.0 is a valid ratio.
function mancino_model(N = 100; T = Float64, backend = nothing, kwargs...)
    ALPHA = 5
    BETA  = T(14.0)
    GAMMA = 3

    # SIF parameter computations
    RN    = T(N)
    BETAN = BETA * RN              # 1400.0

    # Compute the scalar A for starting point
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
    # where H[i] = sum_{j≠i} sqrt(i/j) * (sin^5(log(sqrt(i/j))) + cos^5(log(sqrt(i/j))))
    # (Both j<i and j>i terms use the same positive formula per SIF arithmetic)
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
            # SIF: R* for j<i, RM for j>i. Both map to simple multiply (RM X Y Z → X = Y*Z).
            SA    = SIJ^ALPHA   # SIJ^5
            CA    = CIJ^ALPHA   # CIJ^5
            H    += SQI_J * (SA + CA)
        end
        x0[i] = A * (H + CI[i])
    end

    # Ratio data: Rij[N*(i-1)+j] = i/j for all i,j in 1..N
    # For j=i: i/i = 1.0 (we subtract this term to get sum_{j≠i})
    Rij = T[T(i) / T(j) for i in 1:N for j in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)
    ExaModels.@add_par(c, Rij_par, Rij)
    ExaModels.@add_par(c, CI_par, CI)

    # Objective: sum_i (BETAN*x[i] + sum_{j≠i} MANC(x[j]; i/j) - CI[i])^2
    # = sum_i (BETAN*x[i] + sum_{j=1..N} MANC(x[j]; i/j) - MANC(x[i]; 1) - CI[i])^2
    # where MANC(X, r) = sqrt(X^2 + r) * (sin^5(log(sqrt(X^2+r))) + cos^5(log(sqrt(X^2+r))))
    ExaModels.@add_obj(
        c,
        (
            BETAN * x[i]
            + sum(
                let vij = sqrt(x[j]^2 + Rij_par[N*(i-1)+j]);
                    vij * (sin(log(vij))^ALPHA + cos(log(vij))^ALPHA)
                end
                for j in 1:N
            )
            - (let vii = sqrt(x[i]^2 + T(1)); vii * (sin(log(vii))^ALPHA + cos(log(vii))^ALPHA) end)
            - CI_par[i]
        )^2
        for i in 1:N
    )

    return ExaModels.ExaModel(c; kwargs...)
end
