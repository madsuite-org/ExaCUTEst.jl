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
# Constraints in CUTEst sequential order j=1..NC:
#   j≡1 mod 3: x[j] + x[j+1]^2 + x[j+2]^2 - 3 = 0
#   j≡2 mod 3: x[j] + x[j+2] + x[j+1]^2 - 1 = 0
#              Exception j=NC-1: x[NC+2] used instead of x[j+2]
#   j≡0 mod 3: x[j-2]*x[NC+1] - 1 = 0
#
# CUTEst variable swap: SIF X(NC+2) ↔ CUTEst x[NC+1] (and vice versa):
#   start[NC+1] = -1.0 (SIF X(NC+2) value), start[NC+2] = 1.5 (SIF X(NC+1) value)
#
# Start: x[i] = 2.0 if mod(i,4)==1, 1.5 if mod(i,4)==2,
#               -1.0 if mod(i,4)==3, 0.5 if mod(i,4)==0
#   (with CUTEst swap at positions NC+1, NC+2)
function lukvle12_model(N = 9997; T = Float64, backend = nothing, kwargs...)
    @assert N % 4 == 1 "N must satisfy N ≡ 1 (mod 4) for LUKVLE12"
    NB = (N - 1) ÷ 4  # number of objective group blocks
    NC = 3 * NB        # total number of constraints

    start = T[
        mod(i, 4) == 1 ? T(2) :
        mod(i, 4) == 2 ? T(1.5) :
        mod(i, 4) == 3 ? T(-1) : T(0.5) for i in 1:N
    ]
    # CUTEst swaps positions NC+1 and NC+2
    start[NC+1] = T(-1)
    start[NC+2] = T(1.5)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: J+1=4I-3, J+2=4I-2, J+3=4I-1, J+4=4I, J+5=4I+1
    # CUTEst swap NC+1 ↔ NC+2 affects I_exc = (NC+3)÷4 = 1875 for NC=7497:
    #   At I_exc: J+2=NC+1 → use x[NC+2]; J+3=NC+2 → use x[NC+1]
    #   OBJ1(I_exc): (x[4*I_exc-3] - x[NC+2])^2
    #   OBJ3(I_exc): (x[NC+1] - x[4*I_exc])^4
    I_exc = (NC + 3) ÷ 4

    # OBJ1: exception at I_exc (4I-2=NC+1 → use x[NC+2])
    ExaModels.@add_obj(c, (x[4*I_exc-3] - x[NC+2])^2)
    ExaModels.@add_obj(c, (x[4i-3] - x[4i-2])^2 for i in 1:I_exc-1)
    ExaModels.@add_obj(c, (x[4i-3] - x[4i-2])^2 for i in I_exc+1:NB)

    # OBJ2: (4I-2=NC+1 and 4I-1=NC+2 at I=I_exc, but (x[NC+2]-x[NC+1])^2=(x[NC+1]-x[NC+2])^2)
    ExaModels.@add_obj(c, (x[4i-2] - x[4i-1])^2 for i in 1:NB)

    # OBJ3: exception at I_exc (4I-1=NC+2 → use x[NC+1])
    ExaModels.@add_obj(c, (x[NC+1] - x[4*I_exc])^4)
    ExaModels.@add_obj(c, (x[4i-1] - x[4i])^4 for i in 1:I_exc-1)
    ExaModels.@add_obj(c, (x[4i-1] - x[4i])^4 for i in I_exc+1:NB)

    # OBJ4: no exceptions (4I hits NC+1 only for non-integer I)
    ExaModels.@add_obj(c, (x[4i] - x[4i+1])^4 for i in 1:NB)

    # Constraints: produce in interleaved order j=1..NC.
    # j=1 (j≡1 mod 3): x[1]+x[2]^2+x[3]^2-3
    # j=2 (j≡2 mod 3): x[2]+x[4]+x[3]^2-1
    # j=3..NC: combined formula with selector arrays (i = j-2, so j = i+2 ∈ 3..NC)
    #   q1[i] for j≡1 mod 3: x[i+2]+x[i+3]^2+x[i+4]^2-3
    #   q2[i] for j≡2 mod 3: x[i+2]+x[i+3]^2+x_i2-1 (x_i2=x[i+4] normally, x[NC+2] at j=NC-1)
    #   q3[i] for j≡0 mod 3: x[i]*x[NC+1]-1

    ExaModels.@add_con(c, x[1] + x[2]^2 + x[3]^2 - T(3))
    ExaModels.@add_con(c, x[2] + x[4] + x[3]^2 - T(1))

    # For i=1..NC-2, j=i+2 ∈ 3..NC
    q1_data = T[((i + 2) % 3 == 1) ? T(1) : T(0) for i in 1:NC-2]
    q2_data = T[((i + 2) % 3 == 2) ? T(1) : T(0) for i in 1:NC-2]
    q3_data = T[((i + 2) % 3 == 0) ? T(1) : T(0) for i in 1:NC-2]
    # qe[i]=1 when j=i+2=NC-1, i.e. i=NC-3: at that index, use x[NC+2] instead of x[i+4]
    qe_data = T[(i == NC - 3) ? T(1) : T(0) for i in 1:NC-2]

    ExaModels.@add_par(c, q1, q1_data)
    ExaModels.@add_par(c, q2, q2_data)
    ExaModels.@add_par(c, q3, q3_data)
    ExaModels.@add_par(c, qe, qe_data)

    # j = i+2:
    #   q1: x[i+2]+x[i+3]^2+x[i+4]^2-3
    #   q2: x[i+2]+(x[i+4]+qe*(x[NC+2]-x[i+4]))+x[i+3]^2-1
    #       = x[i+2]+(1-qe)*x[i+4]+qe*x[NC+2]+x[i+3]^2-1
    #       = x[i+2]+x[i+4]-qe*x[i+4]+qe*x[NC+2]+x[i+3]^2-1
    #       = x[i+2]+x[i+4]+qe*(x[NC+2]-x[i+4])+x[i+3]^2-1
    #   q3: x[i]*x[NC+1]-1
    ExaModels.@add_con(c,
        q1[i] * (x[i+2] + x[i+3]^2 + x[i+4]^2 - T(3)) +
        q2[i] * (x[i+2] + x[i+4] + qe[i] * (x[NC+2] - x[i+4]) + x[i+3]^2 - T(1)) +
        q3[i] * (x[i] * x[NC+1] - T(1))
        for i in 1:NC-2)

    return ExaModels.ExaModel(c; kwargs...)
end
