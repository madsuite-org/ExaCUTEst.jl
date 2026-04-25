# BRATU2D: 2D Bratu problem on unit square (P=72, constrained NLS)
# Source: More' (1989), problem 3; SIF input: Ph. Toint, Dec 1989
# classification NOR2-MN-V-V
#
# P^2 variables U(I,J) on P×P grid; inner vars (I,J=2..P-1) are free,
# boundary vars are fixed at 0.
# Parameters: LAMBDA=4.0, H=1/(P-1), C=H^2*LAMBDA
#
# (P-2)^2 equality constraints, one per inner point G(I,J):
#   4*U(I,J) - U(I+1,J) - U(I-1,J) - U(I,J+1) - U(I,J-1) - C*exp(U(I,J)) = 0
#
# CUTEst variable reindexing (complex, P-dependent):
# - Inner vars with SIF pos (J-1)*P+I <= (P-2)^2: keep SIF pos as CUTEst index
# - Inner vars with SIF pos > (P-2)^2: assigned to boundary-slot positions
# - Boundary vars with SIF pos > (P-2)^2: keep SIF pos as CUTEst index
# - Boundary vars with SIF pos <= (P-2)^2: assigned to remaining fixed range
function _bratu2d_var_map(P)
    inner_cutoff = (P - 2)^2
    var_map = zeros(Int, P, P)

    # Inner vars with SIF pos <= inner_cutoff: keep SIF pos
    for J in 2:P-1, I in 2:P-1
        sif = (J - 1) * P + I
        if sif <= inner_cutoff
            var_map[I, J] = sif
        end
    end

    # Boundary holes: boundary SIF positions <= inner_cutoff, sorted descending
    bnd_holes = sort(
        [Int((J - 1) * P + I) for J in 1:P for I in 1:P
         if (I == 1 || I == P || J == 1 || J == P) && (J - 1) * P + I <= inner_cutoff],
        rev = true,
    )

    # Large inner vars (SIF pos > inner_cutoff): assign to bnd_holes in ascending SIF order
    large_inner = sort(
        [(Int((J - 1) * P + I), I, J) for J in 2:P-1 for I in 2:P-1
         if (J - 1) * P + I > inner_cutoff],
        by = x -> x[1],
    )
    for (k, (_, I, J)) in enumerate(large_inner)
        var_map[I, J] = bnd_holes[k]
    end

    # Boundary vars with SIF pos > inner_cutoff: keep SIF pos
    for J in 1:P, I in 1:P
        if (I == 1 || I == P || J == 1 || J == P)
            sif = (J - 1) * P + I
            if sif > inner_cutoff
                var_map[I, J] = sif
            end
        end
    end

    # Boundary vars with SIF pos <= inner_cutoff: assign to remaining fixed-range slots
    bnd_large_sif = Set(
        [Int((J - 1) * P + I) for J in 1:P for I in 1:P
         if (I == 1 || I == P || J == 1 || J == P) && (J - 1) * P + I > inner_cutoff]
    )
    remaining = sort(collect(setdiff(Set(inner_cutoff+1:P^2), bnd_large_sif)))
    bnd_small = sort(
        [Int((J - 1) * P + I) for J in 1:P for I in 1:P
         if (I == 1 || I == P || J == 1 || J == P) && (J - 1) * P + I <= inner_cutoff],
        rev = true,
    )
    for (k, sif) in enumerate(bnd_small)
        Jb = (sif - 1) ÷ P + 1
        Ib = sif - (Jb - 1) * P
        var_map[Ib, Jb] = remaining[k]
    end

    return var_map
end

function bratu2d_model(P = 72; T = Float64, backend = nothing, kwargs...)
    LAMBDA = T(4.0)
    H = T(1) / T(P - 1)
    CC = H^2 * LAMBDA          # C = H^2 * LAMBDA (coefficient for exp term)
    N = P * P
    vm = _bratu2d_var_map(P)

    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf), N)
    for J in 1:P, I in 1:P
        if I == 1 || I == P || J == 1 || J == P
            v = vm[I, J]
            lvar[v] = T(0)
            uvar[v] = T(0)
        end
    end

    # Constraint data: G(I,J) for I=2..P-1, J=2..P-1 (I outer, J inner matches CUTEst ordering)
    con_data = [
        (k = vm[i, j], kip = vm[i+1, j], kim = vm[i-1, j], kjp = vm[i, j+1], kjm = vm[i, j-1])
        for i in 2:P-1 for j in 2:P-1
    ]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, u, N; lvar = lvar, uvar = uvar, start = T(0))
    ExaModels.@add_con(
        c,
        T(4) * u[d.k] - u[d.kip] - u[d.kim] - u[d.kjp] - u[d.kjm] - CC * exp(u[d.k])
        for d in con_data
    )
    return ExaModels.ExaModel(c; kwargs...)
end
