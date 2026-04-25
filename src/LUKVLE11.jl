# LUKVLE11: Chained HS46 problem
# Source: Problem 5.11, Luksan & Vlcek (1999), TR 767
# N=9998 variables (default), all free
# (N-2)/3 = 3332 groups of 4 objective terms each
# NC = 2*(N-2)/3 = 6664 equality constraints (alternating C(K), C(K+1))
#
# With J = 3*(I-1) for I=1..(N-2)/3:
#   OBJ1(I): (x[J+1] - x[J+2])^2                (L2)
#   OBJ2(I): (x[J+3] - 1)^2                      (L2)
#   OBJ3(I): (x[J+4] - 1)^4                      (L4)
#   OBJ4(I): (x[J+5] - 1)^6                      (L6)
#
# Constraints in CUTEst sequential order j=1..NC:
#   j odd:  x[j]^2*x[j+3] + sin(x[j+3] - x[j+4]) - 1 = 0
#           Exception j=NC-1: uses x[2] instead of x[j+4]
#   j even: x[j] + x[j+1]^2*x[j+2] - 2 = 0
#           Exception j=2: uses x[NC+3] instead of x[j=2]
#
# CUTEst swaps SIF variables X(2) and X(NC+3):
#   CUTEst x[2] = SIF X(NC+3) value = 2.0 (mod(NC+3,3)=1 → 2.0)
#   CUTEst x[NC+3] = SIF X(2) value = 1.5 (mod(2,3)=2 → 1.5)
# This swap applies to ALL occurrences in objective and constraints.
#
# Objective exceptions due to swap:
#   OBJ1(I=1):       (x[1] - x[NC+3])^2   (3I-1=2 → x[NC+3])
#   OBJ1(I=(NC+5)/3):(x[2] - x[...])^2    (3I-2=NC+3 → x[2])
#   OBJ3(I=(NC+2)/3):(x[2] - 1)^4         (3I+1=NC+3 → x[2])
#
# Start: x[i] = 2.0 if mod(i,3)==1, 1.5 if mod(i,3)==2, 0.5 if mod(i,3)==0
#   (with CUTEst swap: start[2]=2.0, start[NC+3]=1.5)
function lukvle11_model(N = 9998; T = Float64, backend = nothing, kwargs...)
    @assert N % 3 == 2 "N must satisfy N ≡ 2 (mod 3) for LUKVLE11"
    NB = (N - 2) ÷ 3  # number of objective group blocks
    NC = 2 * NB        # total number of constraints

    start = T[mod(i, 3) == 1 ? T(2) : mod(i, 3) == 2 ? T(1.5) : T(0.5) for i in 1:N]
    # CUTEst swaps positions 2 and NC+3
    start[2]    = T(2)
    start[NC+3] = T(1.5)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: J+1=3I-2, J+2=3I-1, J+3=3I, J+4=3I+1, J+5=3I+2
    # Exception indices:
    #   I_obj1a = 1          (3I-1=2 → use x[NC+3])
    #   I_obj1b = (NC+5)÷3   (3I-2=NC+3 → use x[2])
    #   I_obj3  = (NC+2)÷3   (3I+1=NC+3 → use x[2])
    I_obj1b = (NC + 5) ÷ 3   # = 2223 for N=9998
    I_obj3  = (NC + 2) ÷ 3   # = 2222 for N=9998

    # OBJ1: scalar exceptions at I=1 and I=I_obj1b; generator for remaining i
    ExaModels.@add_obj(c, (x[1] - x[NC+3])^2)                         # I=1 exception
    ExaModels.@add_obj(c, (x[2] - x[3*I_obj1b-1])^2)                  # I=I_obj1b exception
    ExaModels.@add_obj(c, (x[3i-2] - x[3i-1])^2 for i in 2:I_obj1b-1)
    ExaModels.@add_obj(c, (x[3i-2] - x[3i-1])^2 for i in I_obj1b+1:NB)

    # OBJ2: no exceptions
    ExaModels.@add_obj(c, (x[3i] - T(1))^2 for i in 1:NB)

    # OBJ3: scalar exception at I=I_obj3; generator for remaining i
    ExaModels.@add_obj(c, (x[2] - T(1))^4)                            # I=I_obj3 exception
    ExaModels.@add_obj(c, (x[3i+1] - T(1))^4 for i in 1:I_obj3-1)
    ExaModels.@add_obj(c, (x[3i+1] - T(1))^4 for i in I_obj3+1:NB)

    # OBJ4: no exceptions
    ExaModels.@add_obj(c, (x[3i+2] - T(1))^6 for i in 1:NB)

    # Constraints in interleaved order j=1..NC using selector arrays.
    # podd[j]=1 for odd j, peven[j]=1 for even j
    # peNC1[j]=1 only at j=NC-1 (exception: odd uses x[2] instead of x[j+4])
    # pe2[j]=1 only at j=2 (exception: even uses x[NC+3] instead of x[j])
    podd_data  = T[j % 2 == 1 ? T(1) : T(0) for j in 1:NC]
    peven_data = T[j % 2 == 0 ? T(1) : T(0) for j in 1:NC]
    peNC1_data = T[j == NC - 1 ? T(1) : T(0) for j in 1:NC]
    pe2_data   = T[j == 2 ? T(1) : T(0) for j in 1:NC]

    ExaModels.@add_par(c, podd,  podd_data)
    ExaModels.@add_par(c, peven, peven_data)
    ExaModels.@add_par(c, peNC1, peNC1_data)
    ExaModels.@add_par(c, pe2,   pe2_data)

    # Combined formula:
    # odd: x[j]^2*x[j+3] + sin(x[j+3] - x[j+4] + peNC1*(x[j+4]-x[2])) - 1
    #   At j=NC-1: peNC1=1 → sin(x[j+3] - x[2]) ✓
    # even: x[j] + pe2*(x[NC+3]-x[j]) + x[j+1]^2*x[j+2] - 2
    #   At j=2: pe2=1 → x[j] + (x[NC+3]-x[j]) = x[NC+3] ✓
    ExaModels.@add_con(c,
        podd[j]  * (x[j]^2 * x[j+3] + sin(x[j+3] - x[j+4] + peNC1[j] * (x[j+4] - x[2])) - T(1)) +
        peven[j] * (x[j] + pe2[j] * (x[NC+3] - x[j]) + x[j+1]^2 * x[j+2] - T(2))
        for j in 1:NC)

    return ExaModels.ExaModel(c; kwargs...)
end
