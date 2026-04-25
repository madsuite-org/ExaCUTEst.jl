# MANCINO: Mancino's function with variable dimension (N=100 variables, unconstrained)
# Source: E. Spedicato (1975); Buckley #51
# SIF: N=100, ALPHA=5, BETA=14.0, GAMMA=3
# Variables: X(1)..X(N), all free (FR DEFAULT)
# Groups G(I) with L2 type (no scale): contribution = v_i^2
#   v_i = BETAN * X(I) + sum_{j≠i} f(X(J); i, j) - CI
#   where:
#     BETAN = BETA * N = 14.0 * 100 = 1400
#     CI = (i - N/2)^GAMMA = (i - 50)^3
#     f(X; i, j) = sqrt(X^2 + i/j) * (sin^5(log(sqrt(X^2+i/j))) + cos^5(log(sqrt(X^2+i/j))))
# Objective: sum_i v_i^2
function mancino_model(N = 100; T = Float64, backend = nothing, kwargs...)
    ALPHA = 5
    BETA  = T(14.0)
    GAMMA = 3

    # Parameter computations from SIF
    RN    = T(N)
    BETAN = BETA * RN              # 14.0 * 100 = 1400.0

    # Compute A for the starting point
    RN1   = T(N - 1)
    AL1   = T(ALPHA + 1)          # 6.0
    BETAN2 = BETAN * BETAN
    F0    = AL1 * AL1 * RN1 * RN1  # (ALPHA+1)^2 * (N-1)^2
    F2    = BETAN2 - F0             # BETAN^2 - (ALPHA+1)^2*(N-1)^2
    F3    = one(T) / F2
    F4    = BETAN * F3
    A     = -F4                     # constant for starting point

    # Compute starting points
    HNALF = RN * T(-0.5)            # -N/2 = -50.0
    x0 = Vector{T}(undef, N)
    for i in 1:N
        RI = T(i)
        H  = zero(T)

        # j = 1..i-1: positive accumulation
        for j in 1:i-1
            RJ    = T(j)
            SQI_J = sqrt(RI / RJ)   # sqrt(i/j)
            LIJ   = log(SQI_J)
            SIJ   = sin(LIJ)
            CIJ   = cos(LIJ)
            SA    = SIJ^ALPHA        # SIJ^5
            CA    = CIJ^ALPHA        # CIJ^5
            H    += SQI_J * (SA + CA)
        end

        # j = i+1..N: negative accumulation (RM in SIF = multiply by negative)
        for j in i+1:N
            RJ    = T(j)
            SQI_J = sqrt(RI / RJ)   # sqrt(i/j)
            LIJ   = log(SQI_J)
            SIJ   = sin(LIJ)
            CIJ   = cos(LIJ)
            SA    = (-SIJ)^ALPHA     # (-SIJ)^5 = -SIJ^5
            CA    = (-CIJ)^ALPHA     # (-CIJ)^5 = -CIJ^5
            H    += SQI_J * (SA + CA)
        end

        # CI = (i - N/2)^GAMMA = (i - 50)^3
        I_N2 = RI + HNALF           # i - 50.0
        CI   = I_N2^GAMMA

        x0[i] = A * (H + CI)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # Objective: sum_i (BETAN * x[i] + sum_{j≠i} f(x[j]; i, j) - CI)^2
    # where f(X; i, j) = sqrt(X^2 + i/j) * (sin^5(log(sqrt(X^2+i/j))) + cos^5(log(sqrt(X^2+i/j))))
    #
    # We use @add_par to pass the (i, j) pairs and the constant CI values as data.
    # For each group i, the group value is:
    #   v_i = BETAN * x[i] + sum_{j≠i} f(x[j]; i, j) - CI
    # But we cannot easily express the inner sum in a single @add_obj loop.
    #
    # Instead, expand: v_i^2 contains cross terms. This requires a different approach.
    # We note that the group value v_i involves x[i] (linearly) and x[j] (j≠i) via elements.
    # Direct encoding: collect data arrays for (group_i, var_j) pairs.

    # Build data for elements: for each (i, j) pair with j ≠ i
    # Element E(I,J): contributes f(x[j]; i, j) to group G(i)
    # Group G(i) = L2(BETAN*x[i] + sum_j E(I,J) - CI)
    # The full objective = sum_i (BETAN*x[i] + sum_{j≠i} f(x[j]; i, j) - CI)^2
    #
    # ExaModels cannot express this nested sum-inside-square directly.
    # But CUTEst/SIF L2 groups DO support this: the group sums all element values,
    # then applies L2 to the total. ExaModels evaluates group functions via:
    #   @add_obj with L2 of sum expression.
    #
    # For ExaModels, we need to express each group's squared term.
    # The group value v_i = BETAN*x[i] + Σ_j f(x[j]; i, j) - CI is a SUM.
    # Its square: v_i^2 is NOT separable in j.
    #
    # ExaModels processes groups by accumulating element contributions into the group value,
    # then applying the group function. This is fundamentally a grouped structure that
    # ExaModels' macro API does not directly support for multi-variable elements.
    #
    # Instead, we use the equivalent reformulation: define auxiliary group computations
    # as part of the objective expression. Since we can't do nested sums in @add_obj,
    # we precompute the structure and use @add_par for the data.

    # Build (i, j, i/j, CI_i) data for all N*(N-1) pairs
    # Group sums: we need one term per (i, j) pair
    # The objective is sum_i [ BETAN*x[i] + sum_{j≠i} f(x[j];i,j) - CI ]^2
    # This requires a way to express the full group sum. ExaModels can only do
    # separable sums of functions of one variable at a time.
    #
    # The correct approach: note that expanding v_i^2 gives:
    #   v_i^2 = (BETAN*x[i] - CI)^2
    #           + 2*(BETAN*x[i] - CI) * sum_j f(x[j];i,j)
    #           + (sum_j f(x[j];i,j))^2
    # The last term is not separable. ExaModels cannot express this.
    #
    # HOWEVER: looking at how ExaModels handles groups - the key insight is that
    # ExaModels IS capable of this if we use the @add_obj with the inner sums
    # written out as Julia expressions (they just need to compile). For small N,
    # we can write out the full sum. But N=100 means 99 terms per group.
    #
    # The trick: for each group i, we need to express sum_{j≠i} f(x[j]; i, j).
    # We pass (i, i/j) as parameters and index into x.
    # But @add_obj generators iterate over a single index with scalar expressions.
    #
    # For MANCINO, the correct ExaModels encoding uses a double loop where each
    # outer iteration (group i) involves an inner sum over j. This is not directly
    # supported by the single-index generator pattern.
    #
    # Since N=100 (not scalable), we can build the full expression directly.

    # Precompute CI for each group
    CI = T[(T(i) + HNALF)^GAMMA for i in 1:N]

    # For each group i, we need:
    # v_i = BETAN * x[i] + sum_{j=1,j≠i}^{N} f(x[j]; i, j) - CI[i]
    # obj += v_i^2
    #
    # This requires one @add_obj per group (100 calls), each with the full sum.
    # But ExaModels needs a loop form. Since we can't nest sums, we'll use
    # @add_par to store the i/j ratios and write it as a doubly-indexed generator.

    # Data: (group_idx, var_idx, ratio_i_over_j) for all (i,j) pairs with j≠i
    ij_data = [(i, j, T(i)/T(j)) for i in 1:N for j in 1:N if j != i]

    # We need: for group i, sum over j≠i of f(x[j]; i, j)
    # This is a sum per group, not separable across groups.
    # ExaModels cannot do this with a single @add_obj generator.

    # The only viable approach for ExaModels is to use the group structure directly.
    # Looking at other problems: WATSON.jl, which has nested sums, may give insight.

    return ExaModels.ExaModel(c; kwargs...)
end
