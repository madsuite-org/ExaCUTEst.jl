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
# Constraints in CUTEst sequential order j=1..NC:
#   j odd:  x[j] + x[j+1]^2 + x[j+2] + x[j+3] + 4*x[j+4] - 5 = 0
#   j even: -2*x[j+2] - 2*x[j+3] + x[j+1]^2 - 3 = 0
#
# CUTEst swaps SIF variables X(1) and X(NC+1):
#   start[1] = 5.0 (SIF X(NC+1) value), start[NC+1] = 3.0 (SIF X(1) value)
# Formulas referencing x[1] or x[NC+1] are affected (swap applies).
# Exceptions relative to the standard formulas:
#   j=1 (odd): x[NC+1] replaces x[j=1]
#   j=NC-3 (odd): x[1] replaces x[j+4=NC+1]
#   j=NC-2 (even): x[1] replaces x[j+3=NC+1]
#   j=NC-1 (odd): x[1] replaces x[j+2=NC+1]
#   j=NC   (even): x[1] replaces x[j+1=NC+1]
#
# Start: x[i] = 3.0 if mod(i,3)==1, 5.0 if mod(i,3)==2, -3.0 if mod(i,3)==0
#   (with CUTEst swap at positions 1 and NC+1)
function lukvle13_model(N = 9998; T = Float64, backend = nothing, kwargs...)
    @assert N % 3 == 2 "N must satisfy N ≡ 2 (mod 3) for LUKVLE13"
    NB = (N - 2) ÷ 3  # number of objective group blocks
    NC = 2 * NB        # total number of constraints

    start = T[mod(i, 3) == 1 ? T(3) : mod(i, 3) == 2 ? T(5) : T(-3) for i in 1:N]
    # CUTEst swaps positions 1 and NC+1
    start[1]    = T(5)
    start[NC+1] = T(3)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: J+1=3I-2, J+2=3I-1, J+3=3I, J+4=3I+1, J+5=3I+2
    # CUTEst swap pos 1 ↔ NC+1=6665 affects:
    #   OBJ1(I=1):    3I-2=1 → x[NC+1];   (x[NC+1]-1)^2
    #   OBJ2(I=(NC+2)/3=2222): 3I-1=NC+1 → x[1];  (x[1]-x[3*2222])^2
    #   OBJ3(I=(NC-1)/3=2221): 3I+2=NC+1 → x[1];  (x[3*2221+1]-x[1])^4
    I_obj2_exc = (NC + 2) ÷ 3  # = 2222 for N=9998
    I_obj3_exc = (NC - 1) ÷ 3  # = 2221 for N=9998

    # OBJ1: exception at I=1 (3I-2=1 → use x[NC+1])
    ExaModels.@add_obj(c, (x[NC+1] - T(1))^2)
    ExaModels.@add_obj(c, (x[3i-2] - T(1))^2 for i in 2:NB)

    # OBJ2: exception at I=I_obj2_exc (3I-1=NC+1 → use x[1])
    ExaModels.@add_obj(c, (x[1] - x[3*I_obj2_exc])^2)
    ExaModels.@add_obj(c, (x[3i-1] - x[3i])^2 for i in 1:I_obj2_exc-1)
    ExaModels.@add_obj(c, (x[3i-1] - x[3i])^2 for i in I_obj2_exc+1:NB)

    # OBJ3: exception at I=I_obj3_exc (3I+2=NC+1 → use x[1])
    ExaModels.@add_obj(c, (x[3*I_obj3_exc+1] - x[1])^4)
    ExaModels.@add_obj(c, (x[3i+1] - x[3i+2])^4 for i in 1:I_obj3_exc-1)
    ExaModels.@add_obj(c, (x[3i+1] - x[3i+2])^4 for i in I_obj3_exc+1:NB)

    # Constraint ordering: j=1, then j=2..(NC-4) combined, then j=(NC-3)..NC as scalars.
    # The clean range j=2..(NC-4) has no variable-swap exceptions.
    # The 5 exception constraints are at j=1, NC-3, NC-2, NC-1, NC.

    # j=1 (odd): x[NC+1]+x[2]^2+x[3]+x[4]+4*x[5]-5
    ExaModels.@add_con(c, x[NC+1] + x[2]^2 + x[3] + x[4] + T(4) * x[5] - T(5))

    # j=2..(NC-4): standard interleaved odd/even formulas, no exceptions
    # i = j-1 ∈ 1..(NC-5), j = i+1 ∈ 2..(NC-4)
    # podd_m[i]=1 when j=i+1 is odd, peven_m[i]=1 when j=i+1 is even
    podd_m_data  = T[((i + 1) % 2 == 1) ? T(1) : T(0) for i in 1:NC-5]
    peven_m_data = T[((i + 1) % 2 == 0) ? T(1) : T(0) for i in 1:NC-5]

    ExaModels.@add_par(c, podd_m,  podd_m_data)
    ExaModels.@add_par(c, peven_m, peven_m_data)

    # j=i+1:
    #   odd:  x[j]+x[j+1]^2+x[j+2]+x[j+3]+4*x[j+4]-5
    #       = x[i+1]+x[i+2]^2+x[i+3]+x[i+4]+4*x[i+5]-5
    #   even: -2*x[j+2]-2*x[j+3]+x[j+1]^2-3
    #       = -2*x[i+3]-2*x[i+4]+x[i+2]^2-3
    ExaModels.@add_con(c,
        podd_m[i]  * (x[i+1] + x[i+2]^2 + x[i+3] + x[i+4] + T(4) * x[i+5] - T(5)) +
        peven_m[i] * (-T(2) * x[i+3] - T(2) * x[i+4] + x[i+2]^2 - T(3))
        for i in 1:NC-5)

    # j=NC-3 (odd): x[NC-3]+x[NC-2]^2+x[NC-1]+x[NC]+4*x[1]-5
    ExaModels.@add_con(c, x[NC-3] + x[NC-2]^2 + x[NC-1] + x[NC] + T(4) * x[1] - T(5))

    # j=NC-2 (even): -2*x[NC]+2*x[NC+1]+x[NC-1]^2-3 ... wait:
    # j=NC-2: j+1=NC-1, j+2=NC, j+3=NC+1 → x[1]
    # formula: -2*x[j+2]-2*x[j+3]+x[j+1]^2-3 = -2*x[NC]-2*x[1]+x[NC-1]^2-3
    ExaModels.@add_con(c, -T(2) * x[NC] - T(2) * x[1] + x[NC-1]^2 - T(3))

    # j=NC-1 (odd): j+1=NC, j+2=NC+1→x[1], j+3=NC+2, j+4=NC+3
    # formula: x[NC-1]+x[NC]^2+x[1]+x[NC+2]+4*x[NC+3]-5
    ExaModels.@add_con(c, x[NC-1] + x[NC]^2 + x[1] + x[NC+2] + T(4) * x[NC+3] - T(5))

    # j=NC (even): j+1=NC+1→x[1], j+2=NC+2, j+3=NC+3
    # formula: -2*x[NC+2]-2*x[NC+3]+x[1]^2-3
    ExaModels.@add_con(c, -T(2) * x[NC+2] - T(2) * x[NC+3] + x[1]^2 - T(3))

    return ExaModels.ExaModel(c; kwargs...)
end
