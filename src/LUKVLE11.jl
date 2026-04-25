# LUKVLE11: Chained HS46 problem
# Source: Problem 5.11, Luksan & Vlcek (1999), TR 767
# N=9998 variables (default), all free
# (N-2)/3 = 3332 groups of 4 objective terms each
# NC = 2*(N-2)/3 = 6664 equality constraints (pairs C(K), C(K+1) for K=1,3,...)
#
# With J = 3*(I-1) for I=1..(N-2)/3:
#   OBJ1(I): (x[J+1] - x[J+2])^2                (L2)
#   OBJ2(I): (x[J+3] - 1)^2                      (L2)
#   OBJ3(I): (x[J+4] - 1)^4                      (L4)
#   OBJ4(I): (x[J+5] - 1)^6                      (L6)
#
# ELEMENT TYPES:
#   C21(V,W) = V^2 * W
#   S(V,W): IV D=V-W, F=sin(D) = sin(V-W)
#
# For K=1,3,...,NC-1 (step 2):
#   C(K):   C21(X(K), X(K+3)) + S(X(K+3), X(K+4)) - 1 = 0
#           i.e., X(K)^2*X(K+3) + sin(X(K+3)-X(K+4)) - 1 = 0
#   C(K+1): X(K+1) + C21(X(K+2), X(K+3)) - 2 = 0
#           i.e., X(K+1) + X(K+2)^2*X(K+3) - 2 = 0
#
# Start: x[i] = 2.0 if mod(i,3)==1, 1.5 if mod(i,3)==2, 0.5 if mod(i,3)==0
function lukvle11_model(N = 9998; T = Float64, backend = nothing, kwargs...)
    @assert N % 3 == 2 "N must satisfy N ≡ 2 (mod 3) for LUKVLE11"
    NB = (N - 2) ÷ 3  # number of objective group blocks
    NC = 2 * NB        # total number of constraints

    start = T[mod(i, 3) == 1 ? T(2) : mod(i, 3) == 2 ? T(1.5) : T(0.5) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: with J = 3*(I-1), i.e., J+1 = 3I-2, J+2 = 3I-1, J+3 = 3I, J+4 = 3I+1, J+5 = 3I+2
    # For I=1..NB:  J+1 = 3I-2, J+2 = 3I-1, J+3 = 3I, J+4 = 3I+1, J+5 = 3I+2
    ExaModels.@add_obj(c, (x[3i-2] - x[3i-1])^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[3i] - T(1))^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[3i+1] - T(1))^4 for i in 1:NB)
    ExaModels.@add_obj(c, (x[3i+2] - T(1))^6 for i in 1:NB)

    # Constraints: K=1,3,...,NC-1 (step 2)
    # C(K):   X(K)^2*X(K+3) + sin(X(K+3)-X(K+4)) - 1 = 0
    # C(K+1): X(K+1) + X(K+2)^2*X(K+3) - 2 = 0
    # K=1,3,5,... means K = 2k-1 for k=1..NB
    ExaModels.@add_con(
        c,
        x[2k-1]^2 * x[2k+2] + sin(x[2k+2] - x[2k+3]) - T(1) for k in 1:NB
    )
    ExaModels.@add_con(c, x[2k] + x[2k+1]^2 * x[2k+2] - T(2) for k in 1:NB)

    return ExaModels.ExaModel(c; kwargs...)
end
