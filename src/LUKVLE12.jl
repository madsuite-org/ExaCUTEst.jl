# LUKVLE12: Chained HS47 problem
# Source: Problem 5.12, Luksan & Vlcek (1999), TR 767
# N=9997 variables (default), all free
# (N-1)/4 = 2499 blocks, NC = 3*(N-1)/4 = 7497 constraints
#
# With J = 4*(I-1) for I=1..(N-1)/4:
#   OBJ1(I): (x[J+1] - x[J+2])^2                (L2)
#   OBJ2(I): (x[J+2] - x[J+3])^2                (L2)
#   OBJ3(I): (x[J+3] - x[J+4])^4                (L4)
#   OBJ4(I): (x[J+4] - x[J+5])^4                (L4)
#
# ELEMENT TYPES:
#   SQR(V) = V^2
#   PROD(V,W) = V*W
#
# For K=1,4,...,NC-2 (step 3), with K+1=K+1, K+2=K+2, K+3=K+3, K+4=K+4:
#   C(K):   x[K] + x[K+1]^2 + x[K+2]^2 - 3 = 0
#   C(K+1): x[K+1] + x[K+3] + x[K+2]^2 - 1 = 0
#   C(K+2): x[K]*x[K+4] - 1 = 0
#
# Start: x[i] = 2.0 if mod(i,4)==1, 1.5 if mod(i,4)==2,
#               -1.0 if mod(i,4)==3, 0.5 if mod(i,4)==0
function lukvle12_model(N = 9997; T = Float64, backend = nothing, kwargs...)
    @assert N % 4 == 1 "N must satisfy N ≡ 1 (mod 4) for LUKVLE12"
    NB = (N - 1) ÷ 4  # number of objective group blocks
    NC = 3 * NB        # total number of constraints

    start = T[
        mod(i, 4) == 1 ? T(2) :
        mod(i, 4) == 2 ? T(1.5) :
        mod(i, 4) == 3 ? T(-1) : T(0.5) for i in 1:N
    ]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: with J+1=4I-3, J+2=4I-2, J+3=4I-1, J+4=4I, J+5=4I+1
    ExaModels.@add_obj(c, (x[4i-3] - x[4i-2])^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i-2] - x[4i-1])^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i-1] - x[4i])^4 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i] - x[4i+1])^4 for i in 1:NB)

    # Constraints: K=1,4,...,NC-2 (step 3), K = 3m-2 for m=1..NB
    # K=3m-2, K+1=3m-1, K+2=3m, K+3=3m+1, K+4=3m+2 are the constraint indices
    # The variable indices in the elements are also K,K+1,...,K+4
    # C(K):   x[K] + x[K+1]^2 + x[K+2]^2 - 3 = 0
    # C(K+1): x[K+1] + x[K+3] + x[K+2]^2 - 1 = 0
    # C(K+2): x[K]*x[K+4] - 1 = 0
    # With K=3m-2: x variables at indices K=3m-2, K+1=3m-1, K+2=3m, K+3=3m+1, K+4=3m+2
    ExaModels.@add_con(
        c,
        x[3m-2] + x[3m-1]^2 + x[3m]^2 - T(3) for m in 1:NB
    )
    ExaModels.@add_con(c, x[3m-1] + x[3m+1] + x[3m]^2 - T(1) for m in 1:NB)
    ExaModels.@add_con(c, x[3m-2] * x[3m+2] - T(1) for m in 1:NB)

    return ExaModels.ExaModel(c; kwargs...)
end
