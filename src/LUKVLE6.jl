# LUKVLE6: Generalized Broyden banded function with exponential constraints
# Source: Problem 5.6, Luksan & Vlcek (1999), TR 767
# N=9999 variables (default), N/2=4999 equality constraints
# All variables free; start x[i]=3
#
# Objective: sum_{i=1}^{N} |v_i|^(7/3)
#   v_i = 2*x[i] + sum_{j=max(1,i-5)}^{min(i+1,N)} x[j] + 5*x[i]^3
#         + sum_{j=max(1,i-5)}^{min(i+1,N)} x[j]^2 + 1
#
# Constraints C(k) for k=1..N/2 (equality = 0):
#   4*x[2k] - (x[2k-1]-x[2k+1])*exp(x[2k-1]-x[2k+1]-x[2k]) - 3 = 0
#
# GROUP TYPE L7/3: F(v) = |v|^(7/3)
# ELEMENT XEXP with IV: U=VM-VP, W=VM-VP-V; F=U*exp(W)=(VM-VP)*exp(VM-VP-V)
function lukvle6_model(N = 9999; T = Float64, backend = nothing, kwargs...)
    N2 = N ÷ 2  # integer division; for XEXP constraints k=1..N/2

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(3))

    # Objective: split into boundary regions to keep fixed structure per generator
    # Region 1: i=1 (band = [1, min(2,N)] = [1,2])
    ExaModels.@add_obj(
        c,
        abs(T(2) * x[1] + x[1] + x[2] + T(5) * x[1]^3 + x[1]^2 + x[2]^2 + T(1))^T(7 / 3)
    )
    # Region 2: i=2 (band = [1, min(3,N)] = [1,3], width=3)
    ExaModels.@add_obj(
        c,
        abs(T(2) * x[2] + (x[1] + x[2] + x[3]) + T(5) * x[2]^3 +
            (x[1]^2 + x[2]^2 + x[3]^2) + T(1))^T(7 / 3)
    )
    # Region 3: i=3 (band = [1,4], width=4)
    if N >= 4
        ExaModels.@add_obj(
            c,
            abs(T(2) * x[3] + (x[1] + x[2] + x[3] + x[4]) + T(5) * x[3]^3 +
                (x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2) + T(1))^T(7 / 3)
        )
    end
    # Region 4: i=4 (band = [1,5], width=5)
    if N >= 5
        ExaModels.@add_obj(
            c,
            abs(T(2) * x[4] + (x[1] + x[2] + x[3] + x[4] + x[5]) + T(5) * x[4]^3 +
                (x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2) + T(1))^T(7 / 3)
        )
    end
    # Region 5: i=5 (band = [1,6], width=6)
    if N >= 6
        ExaModels.@add_obj(
            c,
            abs(T(2) * x[5] + (x[1] + x[2] + x[3] + x[4] + x[5] + x[6]) +
                T(5) * x[5]^3 +
                (x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2 + x[6]^2) +
                T(1))^T(7 / 3)
        )
    end
    # Interior: i=6..N-1 (band = [i-5, i+1], width=7, fully interior)
    if N >= 7
        ExaModels.@add_obj(
            c,
            abs(T(2) * x[i] +
                (x[i-5] + x[i-4] + x[i-3] + x[i-2] + x[i-1] + x[i] + x[i+1]) +
                T(5) * x[i]^3 +
                (x[i-5]^2 + x[i-4]^2 + x[i-3]^2 + x[i-2]^2 + x[i-1]^2 + x[i]^2 + x[i+1]^2) +
                T(1))^T(7 / 3) for i in 6:(N-1)
        )
    end
    # Right boundary: i=N (band = [N-5, N], width=6)
    ExaModels.@add_obj(
        c,
        abs(T(2) * x[N] +
            (x[N-5] + x[N-4] + x[N-3] + x[N-2] + x[N-1] + x[N]) +
            T(5) * x[N]^3 +
            (x[N-5]^2 + x[N-4]^2 + x[N-3]^2 + x[N-2]^2 + x[N-1]^2 + x[N]^2) +
            T(1))^T(7 / 3)
    )

    # Constraints C(k): 4*x[2k] - (x[2k-1]-x[2k+1])*exp(x[2k-1]-x[2k+1]-x[2k]) - 3 = 0
    ExaModels.@add_con(
        c,
        T(4) * x[2k] - (x[2k-1] - x[2k+1]) * exp(x[2k-1] - x[2k+1] - x[2k]) - T(3)
        for k in 1:N2
    )

    return ExaModels.ExaModel(c; kwargs...)
end
