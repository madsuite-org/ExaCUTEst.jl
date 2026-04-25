# LUKVLE15: Chained modified HS50 problem
# Source: Problem 5.15, Luksan & Vlcek (1999), TR 767
# N=9997 variables (default), all free
# (N-1)/4 = 2499 blocks, NC = 3*(N-1)/4 = 7497 constraints
#
# With J = 4*(I-1) for I=1..(N-1)/4:
#   OBJ1(I): (x[J+1] - x[J+2])^2                (L2)
#   OBJ2(I): (x[J+2] - x[J+3])^2                (L2)
#   OBJ3(I): (x[J+3] - x[J+4])^4                (L4)
#   OBJ4(I): (x[J+4] - x[J+5])^4                (L4)
#
# Constraints: C(j) for j=1..NC (sequential, each constraint C(j) uses x[j..j+2]):
#   C(j): x[j]^2 + 2*x[j+1] + 3*x[j+2] - 6 = 0
#
# Note: The 3 SIF types C(K), C(K+1), C(K+2) for K=3m-2 all reduce to
# the same formula x[j]^2 + 2*x[j+1] + 3*x[j+2] - 6 in terms of j.
#
# Start: x[i] = 35.0 if mod(i,4)==1, 11.0 if mod(i,4)==2,
#               5.0 if mod(i,4)==3, -5.0 if mod(i,4)==0
function lukvle15_model(N = 9997; T = Float64, backend = nothing, kwargs...)
    @assert N % 4 == 1 "N must satisfy N ≡ 1 (mod 4) for LUKVLE15"
    NB = (N - 1) ÷ 4  # number of objective group blocks
    NC = 3 * NB        # total number of constraints

    start = T[
        mod(i, 4) == 1 ? T(35) :
        mod(i, 4) == 2 ? T(11) :
        mod(i, 4) == 3 ? T(5) : T(-5) for i in 1:N
    ]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: with J+1=4I-3, J+2=4I-2, J+3=4I-1, J+4=4I, J+5=4I+1
    ExaModels.@add_obj(c, (x[4i-3] - x[4i-2])^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i-2] - x[4i-1])^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i-1] - x[4i])^4 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i] - x[4i+1])^4 for i in 1:NB)

    # Constraints: single formula C(j) = x[j]^2 + 2*x[j+1] + 3*x[j+2] - 6 for j=1..NC
    # This is equivalent to the 3-type SIF structure since all types reduce to this.
    ExaModels.@add_con(c, x[j]^2 + T(2) * x[j+1] + T(3) * x[j+2] - T(6) for j in 1:NC)

    return ExaModels.ExaModel(c; kwargs...)
end
