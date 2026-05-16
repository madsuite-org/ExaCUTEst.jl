# SPIN2: Condensed particle spin equilibrium (no V variables), N=50 default
# Source: N. Trefethen; SIF input by N. Gould, 2009
# classification NOR2-AN-V-V (condensed version of SPIN without auxiliary V variables)
#
# CUTEst variable ordering (interleaved X/Y after MU, OMEGA):
#   MU=x[1], OMEGA=x[2], X(I)=x[2*I+1], Y(I)=x[2*I+2] for I=1..N
#
# Constraints (E type, SIF order: R then I):
#   R(I): -MU*X(I) + OMEGA*Y(I) + sum_{K≠I} (Y(I)-Y(K))/((Y(I)-Y(K))^2+(X(I)-X(K))^2) = 0
#   I(I): -MU*Y(I) - OMEGA*X(I) - sum_{K≠I} (X(I)-X(K))/((X(I)-X(K))^2+(Y(I)-Y(K))^2) = 0
function spin2_model(N = 50; T = Float64, backend = nothing, kwargs...)
    PI     = T(4) * atan(T(1))
    TWO_PI_N = T(2) * PI / T(N)

    nv = 2 + 2 * N
    startv = fill(T(1), nv)
    for I in 1:N
        startv[2*I + 1] = cos(TWO_PI_N * T(I))
        startv[2*I + 2] = sin(TWO_PI_N * T(I))
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = T(-Inf), uvar = T(Inf), start = startv)

    # R(I) constraints (all N), then I(I) constraints (all N) — CUTEst order
    for I in 1:N
        others = [k for k in 1:N if k != I]
        ExaModels.@add_con(c,
            -x[1] * x[2*I + 1] + x[2] * x[2*I + 2] +
            sum((x[2*I + 2] - x[2*k + 2]) /
                ((x[2*I + 2] - x[2*k + 2])^2 + (x[2*I + 1] - x[2*k + 1])^2)
                for k in others))
    end
    for I in 1:N
        others = [k for k in 1:N if k != I]
        ExaModels.@add_con(c,
            -x[1] * x[2*I + 2] - x[2] * x[2*I + 1] -
            sum((x[2*I + 1] - x[2*k + 1]) /
                ((x[2*I + 1] - x[2*k + 1])^2 + (x[2*I + 2] - x[2*k + 2])^2)
                for k in others))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
