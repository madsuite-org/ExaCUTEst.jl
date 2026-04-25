# BRATU2DT: 2D Bratu problem at turning point (P=72, constrained NLS)
# Source: More' (1989), problem 3; SIF input: Ph. Toint, Dec 1989
# classification NOR2-MN-V-V
#
# Identical structure to BRATU2D, with LAMBDA=6.80812 (branching point value).
# Start: all variables at 0 (XV 'DEFAULT' 0.0).
# See BRATU2D.jl for full documentation of the model structure.
function bratu2dt_model(P = 72; T = Float64, backend = nothing, kwargs...)
    LAMBDA = T(6.80812)
    H = T(1) / T(P - 1)
    CC = H^2 * LAMBDA          # C = H^2 * LAMBDA
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
