# LUKVLE13: Chained modified HS48 problem
# Source: Problem 5.13, Luksan & Vlcek (1999), TR 767
# N=9998 variables (default), all free
# (N-2)/3 = 3332 blocks, NC = 2*(N-2)/3 = 6664 constraints
#
# With J = 3*(I-1) for I=1..(N-2)/3:
#   OBJ1(I): (x[J+1] - 1)^2                      (L2)
#   OBJ2(I): (x[J+2] - x[J+3])^2                 (L2)
#   OBJ3(I): (x[J+4] - x[J+5])^4                 (L4)
#
# ELEMENT TYPES:
#   SQR(V) = V^2
#
# For K=1,3,...,NC-1 (step 2), K=2m-1 for m=1..NB:
#   C(K):   x[K] + x[K+1]^2 + x[K+2] + x[K+3] + 4*x[K+4] - 5 = 0
#   C(K+1): -2*x[K+3] - 2*x[K+4] + x[K+2]^2 - 3 = 0
#
# Start: x[i] = 3.0 if mod(i,3)==1, 5.0 if mod(i,3)==2, -3.0 if mod(i,3)==0
function lukvle13_model(N = 9998; T = Float64, backend = nothing, kwargs...)
    @assert N % 3 == 2 "N must satisfy N ≡ 2 (mod 3) for LUKVLE13"
    NB = (N - 2) ÷ 3  # number of objective group blocks
    NC = 2 * NB        # total number of constraints

    start = T[mod(i, 3) == 1 ? T(3) : mod(i, 3) == 2 ? T(5) : T(-3) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: with J+1=3I-2, J+2=3I-1, J+3=3I, J+4=3I+1, J+5=3I+2
    ExaModels.@add_obj(c, (x[3i-2] - T(1))^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[3i-1] - x[3i])^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[3i+1] - x[3i+2])^4 for i in 1:NB)

    # Constraints: K=2m-1 for m=1..NB
    # C(K):   x[K] + x[K+1]^2 + x[K+2] + x[K+3] + 4*x[K+4] - 5 = 0
    # C(K+1): -2*x[K+3] - 2*x[K+4] + x[K+2]^2 - 3 = 0
    # With K=2m-1: K+1=2m, K+2=2m+1, K+3=2m+2, K+4=2m+3
    ExaModels.@add_con(
        c,
        x[2m-1] + x[2m]^2 + x[2m+1] + x[2m+2] + T(4) * x[2m+3] - T(5) for m in 1:NB
    )
    ExaModels.@add_con(
        c,
        -T(2) * x[2m+2] - T(2) * x[2m+3] + x[2m+1]^2 - T(3) for m in 1:NB
    )

    return ExaModels.ExaModel(c; kwargs...)
end
