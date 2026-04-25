# LUKVLE16: Chained modified HS51 problem
# Source: Problem 5.16, Luksan & Vlcek (1999), TR 767
# N=9997 variables (default), all free
# (N-1)/4 = 2499 blocks, NC = 3*(N-1)/4 = 7497 constraints
#
# With J = 4*(I-1) for I=1..(N-1)/4:
#   OBJ1(I): (x[J+1] - x[J+2])^4                (L4)
#   OBJ2(I): (x[J+2] + x[J+3] - 2)^2            (L2)
#   OBJ3(I): (x[J+4] - 1)^2                      (L2)
#   OBJ4(I): (x[J+5] - 1)^2                      (L2)
#
# ELEMENT TYPES:
#   SQR(V) = V^2
#
# For K=1,4,...,NC-2 (step 3), K=3m-2 for m=1..NB:
#   E(K)=SQR(X(K)), E(K+1)=SQR(X(K+2)), E(K+2)=SQR(X(K+1))
#   C(K):   x[K]^2 + 3*x[K+1] - 4 = 0
#   C(K+1): x[K+2]^2 + x[K+3] - 2*x[K+4] = 0
#   C(K+2): x[K+1]^2 - x[K+4] = 0
#
# Start: x[i] = 2.5 if mod(i,4)==1, 0.5 if mod(i,4)==2,
#               2.0 if mod(i,4)==3, -1.0 if mod(i,4)==0
function lukvle16_model(N = 9997; T = Float64, backend = nothing, kwargs...)
    @assert N % 4 == 1 "N must satisfy N ≡ 1 (mod 4) for LUKVLE16"
    NB = (N - 1) ÷ 4  # number of objective group blocks
    NC = 3 * NB        # total number of constraints

    start = T[
        mod(i, 4) == 1 ? T(2.5) :
        mod(i, 4) == 2 ? T(0.5) :
        mod(i, 4) == 3 ? T(2) : T(-1) for i in 1:N
    ]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: with J+1=4I-3, J+2=4I-2, J+3=4I-1, J+4=4I, J+5=4I+1
    ExaModels.@add_obj(c, (x[4i-3] - x[4i-2])^4 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i-2] + x[4i-1] - T(2))^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i] - T(1))^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i+1] - T(1))^2 for i in 1:NB)

    # Constraints: K=3m-2 for m=1..NB
    # E(K)=SQR(X(K)), E(K+1)=SQR(X(K+2)), E(K+2)=SQR(X(K+1))
    # C(K):   3*x[K+1] + x[K]^2 - 4 = 0
    # C(K+1): x[K+3] - 2*x[K+4] + x[K+2]^2 = 0
    # C(K+2): -x[K+4] + x[K+1]^2 = 0
    # With K=3m-2: K+1=3m-1, K+2=3m, K+3=3m+1, K+4=3m+2
    ExaModels.@add_con(c, T(3) * x[3m-1] + x[3m-2]^2 - T(4) for m in 1:NB)
    ExaModels.@add_con(c, x[3m+1] - T(2) * x[3m+2] + x[3m]^2 for m in 1:NB)
    ExaModels.@add_con(c, -x[3m+2] + x[3m-1]^2 for m in 1:NB)

    return ExaModels.ExaModel(c; kwargs...)
end
